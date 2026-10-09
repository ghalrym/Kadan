"""Pinned H3 CPU block0 tail: real scales/norm2/gated FF; no SGLang import.
Usage: COMPONENT H3_SOURCE_DIRECTORY CHECKPOINT QKV_EVIDENCE ATTENTION_EVIDENCE NEW_DIRECTORY
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

SOURCE='14661aa1ef85c345eb1c64139d5640e98eab2dc35cd8b6e75151b51181d032d6'
ATTENTION='5226e44baf40b7f2aefdc65f66f94ad4a611440f68cff098527904808df198ad'
QKV_MANIFEST='570a296a55cf277081495383ab95309fb11142d8dc40ba4a993f73597accb48c'


def run(binary,source,checkpoint,qkv_root,attention_root,output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    data=(source/'base_module.py').read_bytes();assert hashlib.sha256(data).hexdigest()==SOURCE
    tree=ast.parse(data);nodes=[]
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name in ('_scaled_residual_add','_unfused_bias_linear'):nodes.append(n)
        if isinstance(n,ast.ClassDef) and n.name=='FeedForward':
            nodes.extend(m for m in n.body if isinstance(m,ast.FunctionDef) and m.name=='_forward_impl')
    assert len(nodes)==3
    namespace=dict(torch=torch,nn=torch.nn,try_fused_scaled_residual_add_exact=lambda *args:None)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source/'base_module.py'),'exec'),namespace)
    before=checkpoint.stat();weights={};hashes={}
    specs={'scale1':(2048,),'norm2.weight':(2048,),'ff.w1.weight':(16384,2048),'ff.w1.bias':(16384,),
           'ff.w2.weight':(2048,8192),'ff.w2.bias':(2048,),'scale2':(2048,)}
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
    ff=SimpleNamespace(w1=linear('ff.w1',2048,16384),w2=linear('ff.w2',8192,2048),use_gated=True,act_fn=torch.nn.SiLU(),silu_and_mul=None)
    manifest=(qkv_root/'manifest.json').read_bytes();assert hashlib.sha256(manifest).hexdigest()==QKV_MANIFEST
    qkv_files=json.loads(manifest)['files']
    prior=(attention_root/'report.json').read_bytes();assert hashlib.sha256(prior).hexdigest()==ATTENTION
    output.mkdir(parents=True,exist_ok=False);cases=[]
    for case in json.loads(prior)['cases']:
        tokens=case['tokens']
        if tokens>2 or case['axis']!=0:continue
        label=f"{tokens}-{case['pattern']}";name=label+'/input.f32'
        residual=(qkv_root/name).read_bytes();assert hashlib.sha256(residual).hexdigest()==qkv_files[name]
        with (attention_root/(label+'-axis0')/'result.tensor').open('rb') as f:
            assert f.readline()==b'KADAN_H3_ATTENTION_V1\n'
            assert list(map(int,f.readline().split()))==[tokens,2048]
            assert f.readline()==b'F32LE\n';attention=f.read()
        assert hashlib.sha256(attention).hexdigest()==case['output_sha256']
        x=torch.from_numpy(np.frombuffer(residual,dtype='<f4').copy().reshape(tokens,2048))
        a=torch.from_numpy(np.frombuffer(attention,dtype='<f4').copy().reshape(tokens,2048))
        summed=namespace['_scaled_residual_add'](x,a,weights['scale1'])
        norm=F.rms_norm(summed,(2048,),weights['norm2.weight'],eps=1e-5)
        feed=namespace['_forward_impl'](ff,norm)
        expected=namespace['_scaled_residual_add'](summed,feed,weights['scale2']).numpy()
        dest=output/label;dest.mkdir();(dest/'residual.f32').write_bytes(residual);(dest/'attention.f32').write_bytes(attention)
        completed=subprocess.run([str(binary),str(checkpoint.parent),checkpoint.name,str(dest/'residual.f32'),str(dest/'attention.f32'),str(dest/'result.tensor')],check=True,capture_output=True,text=True)
        with (dest/'result.tensor').open('rb') as f:
            assert f.readline()==b'KADAN_H3_FEED_FORWARD_V1\n'
            assert list(map(int,f.readline().split()))==[tokens,2048]
            assert f.readline()==b'F32LE\n';data=f.read()
        actual=np.frombuffer(data,dtype='<f4').reshape(tokens,2048)
        absolute=np.abs(actual-expected);bound=2e-4+2e-4*np.abs(expected)
        assert np.isfinite(actual).all() and np.all(absolute<=bound),(label,float(absolute.max()))
        cases.append(dict(tokens=tokens,pattern=case['pattern'],values=actual.size,max_absolute_error=float(absolute.max()),
            max_tolerance_ratio=float((absolute/bound).max()),residual_sha256=qkv_files[name],attention_sha256=case['output_sha256'],
            output_sha256=hashlib.sha256(data).hexdigest(),stdout=completed.stdout.strip()))
    assert len(cases)==6
    after=checkpoint.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    report=dict(reference='Pinned H3 CPU scaled residuals and FeedForward._forward_impl, Torch RMSNorm F32',source_sha256=SOURCE,
        prior_attention_report_sha256=ATTENTION,prior_qkv_manifest_sha256=QKV_MANIFEST,selected_tensor_sha256=hashes,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),reference_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch_version=torch.__version__,cpu_threads=1,interop_threads=1,cpu_affinity=[cpu],gpu_execution=False,full_video_generation=False,
        production_fp16_parity=False,atol=2e-4,rtol=2e-4,cases=cases,compared_values=sum(c['values'] for c in cases),
        max_absolute_error=max(c['max_absolute_error'] for c in cases),max_tolerance_ratio=max(c['max_tolerance_ratio'] for c in cases))
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))


if __name__=='__main__':run(*(Path(p).resolve() for p in sys.argv[1:]))
