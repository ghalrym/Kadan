"""Pinned complete H3 block0 CPU reference; no native intermediate is reused.
Usage: COMPONENT H3_SOURCE_DIRECTORY CHECKPOINT NEW_OUTPUT_DIRECTORY
Only hash-checked AST methods are executed; no SGLang or GPU package import.
"""
import ast
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PINNED={
    'attention.py':'a2d06ed14251937f98c8e903fb653282236222cc938569a37a1a3bb17b4579b0',
    'vit_utils.py':'fa142186b313ab33d049225d49de44e9f09527203cb1166a2b1f7f02667a9f25',
    'base_module.py':'14661aa1ef85c345eb1c64139d5640e98eab2dc35cd8b6e75151b51181d032d6',
    'vae_vit.py':'1e11a02564f2acbcdaed991a9fcb2e7060815b39c7cc4c56ed1bbcddbddb172d',
}
SELECTED={'vit_utils.py':['_env_flag','_rotate_half','_apply_rotary_pos_emb_impl'],
          'attention.py':['_vit_norm_input','_apply_qk_norm','_sdpa_attention'],
          'base_module.py':['RotaryEmbeddingND','_scaled_residual_add','_unfused_bias_linear']}
METHODS={('base_module.py','TransformerBlock','forward'):'block_forward',
         ('base_module.py','FeedForward','_forward_impl'):'ff_forward',
         ('attention.py','Attention','forward'):'attention_forward'}


def run(binary,source,checkpoint,output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    ns=dict(torch=torch,nn=torch.nn,F=F,math=math,os=os,nullcontext=nullcontext,Optional=Optional,Tuple=Tuple,
            _FORCE_ROCM_MATH_SDPA=False,try_fused_scaled_residual_add_exact=lambda *args:None,
            _try_fused_qk_rmsnorm_rope=lambda *args:None)
    found=set()
    for filename,digest in PINNED.items():
        data=(source/filename).read_bytes();assert hashlib.sha256(data).hexdigest()==digest,filename
        nodes=[]
        for node in ast.parse(data).body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in SELECTED.get(filename,[]):nodes.append(node)
            if isinstance(node,ast.ClassDef):
                for method in node.body:
                    key=(filename,node.name,getattr(method,'name',''))
                    if key in METHODS:
                        method.name=METHODS[key];nodes.append(method);found.add(key)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source/filename),'exec'),ns)
    assert found==set(METHODS)
    ns['apply_rotary_pos_emb_qk']=lambda q,k,table:tuple(ns['_apply_rotary_pos_emb_impl'](v,table) for v in (q,k))
    specs={'norm1.weight':(2048,),'attn.to_qkv.weight':(6144,2048),'attn.to_qkv.bias':(6144,),
           'attn.to_out.weight':(2048,2048),'attn.to_out.bias':(2048,),
           'scale1':(2048,),'norm2.weight':(2048,),'ff.w1.weight':(16384,2048),'ff.w1.bias':(16384,),
           'ff.w2.weight':(2048,8192),'ff.w2.bias':(2048,),'scale2':(2048,)}
    before=checkpoint.stat();weights={};hashes={}
    with checkpoint.open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0];assert length<=1024*1024
        encoded=f.read(length);header=json.loads(encoded)
        assert hashlib.sha256(encoded).hexdigest()=='7bd6afbb1b2cfbf7c19901869eb224ec1b3e2a8cdbd8516d18f515c60f157c26'
        for name,shape in specs.items():
            tensor=header['decoder.transformer_blocks.0.'+name]
            assert tensor['dtype']=='F16' and tuple(tensor['shape'])==shape
            start,end=tensor['data_offsets'];assert end-start==int(np.prod(shape))*2
            assert 0<=start<end<=before.st_size-length-8
            f.seek(8+length+start);data=f.read(end-start);assert len(data)==end-start
            hashes[name]=hashlib.sha256(data).hexdigest()
            weights[name]=torch.from_numpy(np.frombuffer(data,dtype='<f2').astype(np.float32).reshape(shape))
    def linear(name,inputs,outputs):
        layer=torch.nn.Linear(inputs,outputs,device='meta')
        layer.weight=torch.nn.Parameter(weights[name+'.weight'],requires_grad=False)
        layer.bias=torch.nn.Parameter(weights[name+'.bias'],requires_grad=False)
        return layer
    def norm(name):
        layer=torch.nn.RMSNorm(2048,eps=1e-5,device='meta')
        layer.weight=torch.nn.Parameter(weights[name+'.weight'],requires_grad=False);return layer
    attention=SimpleNamespace(to_qkv=linear('attn.to_qkv',2048,6144),to_out=linear('attn.to_out',2048,2048),dim_head=64,
        norm_q=torch.nn.RMSNorm(64,eps=1e-5,elementwise_affine=False),norm_k=torch.nn.RMSNorm(64,eps=1e-5,elementwise_affine=False),attn=None)
    ff=SimpleNamespace(w1=linear('ff.w1',2048,16384),w2=linear('ff.w2',8192,2048),use_gated=True,act_fn=torch.nn.SiLU(),silu_and_mul=None)
    block=SimpleNamespace(norm1=norm('norm1'),norm2=norm('norm2'),use_scale=True,scale1=weights['scale1'],scale2=weights['scale2'],
        attn=lambda x,positions:ns['attention_forward'](attention,x,positions),ff=lambda x:ns['ff_forward'](ff,x))
    rotary=ns['RotaryEmbeddingND'](48,100,n_dim=3,use_angle=True)
    output.mkdir(parents=True,exist_ok=False);cases=[];rng=np.random.default_rng(133)
    for tokens in (1,2):
        for pattern in ('zero','ramp','random'):
            x=np.zeros((1,tokens,2048),dtype=np.float32)
            if pattern=='ramp':x[:]=np.linspace(-1,1,x.size,dtype=np.float32).reshape(x.shape)
            if pattern=='random':x[:]=rng.normal(0,.3,x.shape).astype(np.float32)
            coords=np.array([[[-.5,.25,.75],[.5,-.25,-.75]]],dtype=np.float32)[:,:tokens,:].copy()
            with torch.no_grad():expected=ns['block_forward'](block,torch.from_numpy(x),rotary(torch.from_numpy(coords))).numpy()
            case=output/f'{tokens}-{pattern}';case.mkdir();(case/'input.f32').write_bytes(x.tobytes());(case/'coordinates.f32').write_bytes(coords.tobytes())
            result=subprocess.run([str(binary),str(checkpoint.parent),checkpoint.name,str(case/'input.f32'),str(case/'coordinates.f32'),str(case/'result.tensor')],check=True,capture_output=True,text=True)
            with (case/'result.tensor').open('rb') as f:
                assert f.readline()==b'KADAN_H3_BLOCK_V1\n'
                assert list(map(int,f.readline().split()))==[tokens,2048]
                assert f.readline()==b'F32LE\n';data=f.read()
            actual=np.frombuffer(data,dtype='<f4').reshape(expected.shape)
            absolute=np.abs(actual-expected);bound=2e-4+2e-4*np.abs(expected)
            assert np.isfinite(actual).all() and np.all(absolute<=bound),(tokens,pattern,float(absolute.max()))
            cases.append(dict(tokens=tokens,pattern=pattern,values=actual.size,max_absolute_error=float(absolute.max()),
                max_tolerance_ratio=float((absolute/bound).max()),input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                coordinates_sha256=hashlib.sha256(coords.tobytes()).hexdigest(),output_sha256=hashlib.sha256(data).hexdigest(),stdout=result.stdout.strip()))
    after=checkpoint.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
    report=dict(reference='Pinned TransformerBlock.forward, Attention.forward, FeedForward._forward_impl, CPU F32; accelerator paths disabled',
        source_sha256=PINNED,selected_tensor_sha256=hashes,header_sha256=hashlib.sha256(encoded).hexdigest(),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),reference_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch_version=torch.__version__,cpu_threads=1,interop_threads=1,cpu_affinity=[cpu],gpu_execution=False,
        full_video_generation=False,production_fp16_parity=False,atol=2e-4,rtol=2e-4,cases=cases,
        compared_values=sum(c['values'] for c in cases),max_absolute_error=max(c['max_absolute_error'] for c in cases),
        max_tolerance_ratio=max(c['max_tolerance_ratio'] for c in cases))
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':run(*(Path(p).resolve() for p in sys.argv[1:]))
