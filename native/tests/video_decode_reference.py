"""Independent pinned H3 decoder forward on CPU; complete graph, no native intermediates.
Usage: DECODER H3_SOURCE_DIRECTORY CHECKPOINT NEW_OUTPUT_DIRECTORY
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

from video_block_reference import PINNED, SELECTED, METHODS


def run(binary,source,checkpoint,output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    ns=dict(torch=torch,nn=torch.nn,F=F,math=math,os=os,nullcontext=nullcontext,Optional=Optional,Tuple=Tuple,
            _FORCE_ROCM_MATH_SDPA=False,try_fused_scaled_residual_add_exact=lambda *args:None,
            _try_fused_qk_rmsnorm_rope=lambda *args:None)
    selected={key:list(value) for key,value in SELECTED.items()}
    selected['vit_utils.py']+=['create_token_ids']
    selected['vae_vit.py']=['_pack_tensors_3d','_unpack_tensors_3d','_cuda_autocast_disabled','_linear_with_module_dtype']
    methods={**METHODS,('vae_vit.py','ViT3DDecoder','forward'):'decoder_forward'}
    found=set()
    for filename,digest in PINNED.items():
        data=(source/filename).read_bytes();assert hashlib.sha256(data).hexdigest()==digest,filename
        nodes=[]
        for node in ast.parse(data).body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in selected.get(filename,[]):nodes.append(node)
            if isinstance(node,ast.ClassDef):
                for method in node.body:
                    key=(filename,node.name,getattr(method,'name',''))
                    if key in methods:method.name=methods[key];nodes.append(method);found.add(key)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source/filename),'exec'),ns)
    assert found==set(methods)
    ns['apply_rotary_pos_emb_qk']=lambda q,k,table:tuple(ns['_apply_rotary_pos_emb_impl'](v,table) for v in (q,k))
    # CPU F32 rotary tables are consumed directly by the pinned unfused attention.
    ns['prepare_rotary_pos_emb']=lambda table,dtype:tuple(value.to(dtype=dtype) for value in table)
    before=checkpoint.stat();hashes={}
    f=checkpoint.open('rb');length=struct.unpack('<Q',f.read(8))[0];assert length<=1024*1024
    encoded=f.read(length);header=json.loads(encoded)
    assert hashlib.sha256(encoded).hexdigest()=='7bd6afbb1b2cfbf7c19901869eb224ec1b3e2a8cdbd8516d18f515c60f157c26'
    def weight(name):
        tensor=header[name];assert tensor['dtype']=='F16';shape=tensor['shape'];start,end=tensor['data_offsets']
        assert end-start==math.prod(shape)*2 and 0<=start<end<=before.st_size-length-8
        f.seek(8+length+start);data=f.read(end-start);assert len(data)==end-start
        hashes[name]=hashlib.sha256(data).hexdigest()
        return torch.from_numpy(np.frombuffer(data,dtype='<f2').astype(np.float32).reshape(shape))
    def linear(name):
        matrix=weight(name+'.weight');bias=weight(name+'.bias')
        layer=torch.nn.Linear(matrix.shape[1],matrix.shape[0],device='meta')
        layer.weight=torch.nn.Parameter(matrix,requires_grad=False);layer.bias=torch.nn.Parameter(bias,requires_grad=False)
        return layer
    def norm(name):
        layer=torch.nn.RMSNorm(2048,eps=1e-5,device='meta');layer.weight=torch.nn.Parameter(weight(name+'.weight'),requires_grad=False);return layer
    def blocks(x,positions):
        for i in range(36):
            p=f'decoder.transformer_blocks.{i}.'
            attention=SimpleNamespace(to_qkv=linear(p+'attn.to_qkv'),to_out=linear(p+'attn.to_out'),dim_head=64,
                norm_q=torch.nn.RMSNorm(64,eps=1e-5,elementwise_affine=False),norm_k=torch.nn.RMSNorm(64,eps=1e-5,elementwise_affine=False),attn=None)
            ff=SimpleNamespace(w1=linear(p+'ff.w1'),w2=linear(p+'ff.w2'),use_gated=True,act_fn=torch.nn.SiLU(),silu_and_mul=None)
            block=SimpleNamespace(norm1=norm(p+'norm1'),norm2=norm(p+'norm2'),use_scale=True,scale1=weight(p+'scale1'),scale2=weight(p+'scale2'),
                attn=lambda inputs,pos:ns['attention_forward'](attention,inputs,pos),ff=lambda inputs:ns['ff_forward'](ff,inputs))
            x=ns['block_forward'](block,x,positions)
            del block,ff,attention
        return x
    final_norm=torch.nn.LayerNorm(2048,eps=1e-5,device='meta')
    final_norm.weight=torch.nn.Parameter(weight('decoder.norm_out.weight'),requires_grad=False)
    final_norm.bias=torch.nn.Parameter(weight('decoder.norm_out.bias'),requires_grad=False)
    decoder=SimpleNamespace(config=SimpleNamespace(patch_size=16,patch_size_t=4),num_register_tokens=4,
        x_embedder=linear('decoder.x_embedder'),register_tokens=weight('decoder.register_tokens'),
        training=False,mask_enabled=False,_rotary_pos_emb_cache=None,
        pos_embed=ns['RotaryEmbeddingND'](48,100,n_dim=3,use_angle=True),
        apply_mask_preprocess=lambda hidden,ids,*args:(hidden,ids),apply_mask_postprocess=lambda hidden,*args:hidden,
        forward_transformer_blocks=blocks,norm_out=final_norm,proj_out=linear('decoder.proj_out'))
    mean=weight('latents_mean').reshape(1,24,1,1,1);std=weight('latents_std').reshape(1,24,1,1,1)
    conv=weight('post_quant_conv.weight');bias=weight('post_quant_conv.bias')
    output.mkdir(parents=True,exist_ok=False);cases=[];rng=np.random.default_rng(151)
    for shape,pattern in [((1,1,2),'random'),((2,1,1),'ramp')]:
        t,h,w=shape;raw=rng.normal(0,.3,(t,h,w,24)).astype(np.float32)
        if pattern=='ramp':raw[:]=np.linspace(-1,1,raw.size,dtype=np.float32).reshape(raw.shape)
        x=torch.from_numpy(raw).permute(3,0,1,2).unsqueeze(0).contiguous()
        with torch.no_grad():expected=ns['decoder_forward'](decoder,F.conv3d(x*std+mean,conv,bias)).numpy()[0]
        case=output/f'{t}-{h}-{w}-{pattern}';case.mkdir();(case/'input.f32').write_bytes(raw.tobytes())
        result=subprocess.run([str(binary),str(checkpoint.parent),checkpoint.name,str(case/'input.f32'),str(t),str(h),str(w),str(case/'frames.tensor')],check=True,capture_output=True,text=True)
        with (case/'frames.tensor').open('rb') as stream:
            assert stream.readline()==b'KADAN_H3_FRAMES_V1\n'
            assert list(map(int,stream.readline().split()))==[3,t*4,h*16,w*16]
            assert stream.readline()==b'F32LE\n';data=stream.read()
        actual=np.frombuffer(data,dtype='<f4').reshape(expected.shape)
        absolute=np.abs(actual-expected);bound=1e-3+1e-3*np.abs(expected)
        assert np.isfinite(actual).all() and np.all(absolute<=bound),(shape,float(absolute.max()),float((absolute/bound).max()))
        cases.append(dict(shape=shape,pattern=pattern,values=actual.size,max_absolute_error=float(absolute.max()),max_tolerance_ratio=float((absolute/bound).max()),
            input_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),output_sha256=hashlib.sha256(data).hexdigest(),stdout=result.stdout.strip()))
    f.close();after=checkpoint.stat()
    assert (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
    report=dict(reference='Pinned ViT3DDecoder.forward and complete 36-block CPU F32 graph from independent F16 checkpoint reads',
        source_sha256=PINNED,selected_tensor_sha256=hashes,header_sha256=hashlib.sha256(encoded).hexdigest(),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),reference_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch_version=torch.__version__,cpu_threads=1,cpu_affinity=[cpu],gpu_execution=False,full_video_generation=False,
        production_fp16_parity=False,atol=1e-3,rtol=1e-3,cases=cases)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='selected_tensor_sha256'},indent=2))

if __name__=='__main__':run(*(Path(p).resolve() for p in sys.argv[1:]))
