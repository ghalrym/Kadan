"""Optional pinned H3 CPU SDPA + real to_out reference; no SGLang import.
Usage: COMPONENT H3_SOURCE_DIRECTORY CHECKPOINT RETAINED_ROPE_EVIDENCE NEW_DIRECTORY
"""
import ast
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

PINNED={'attention.py':'a2d06ed14251937f98c8e903fb653282236222cc938569a37a1a3bb17b4579b0',
        'vae_vit.py':'1e11a02564f2acbcdaed991a9fcb2e7060815b39c7cc4c56ed1bbcddbddb172d'}
ROPE_REPORT='186ab0edcdeb98088dc4141c7932578249fa069142837bb2ec1736ce129e0d77'


def run(binary,source,checkpoint,rope_root,output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    namespace=dict(F=F,nullcontext=nullcontext,_FORCE_ROCM_MATH_SDPA=False)
    for filename,digest in PINNED.items():
        data=(source/filename).read_bytes();assert hashlib.sha256(data).hexdigest()==digest
        if filename=='attention.py':
            nodes=[n for n in ast.parse(data).body if isinstance(n,ast.FunctionDef) and n.name=='_sdpa_attention']
            assert len(nodes)==1
            exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source/filename),'exec'),namespace)
    before=checkpoint.stat();weights={};hashes={}
    with checkpoint.open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0];assert length<=1024*1024
        encoded=f.read(length);header=json.loads(encoded)
        assert hashlib.sha256(encoded).hexdigest()=='7bd6afbb1b2cfbf7c19901869eb224ec1b3e2a8cdbd8516d18f515c60f157c26'
        for name,shape in {'weight':(2048,2048),'bias':(2048,)}.items():
            tensor=header['decoder.transformer_blocks.0.attn.to_out.'+name]
            assert tensor['dtype']=='F16' and tuple(tensor['shape'])==shape
            start,end=tensor['data_offsets'];assert end-start==int(np.prod(shape))*2
            assert 0<=start<end<=before.st_size-length-8
            f.seek(8+length+start);data=f.read(end-start);assert len(data)==end-start
            hashes[name]=hashlib.sha256(data).hexdigest()
            weights[name]=torch.from_numpy(np.frombuffer(data,dtype='<f2').astype(np.float32).reshape(shape))
    prior=(rope_root/'report.json').read_bytes();assert hashlib.sha256(prior).hexdigest()==ROPE_REPORT
    upstream=json.loads(prior);assert len(upstream['cases'])==75
    output.mkdir(parents=True,exist_ok=False);cases=[]
    for case in upstream['cases']:
        tokens=case['tokens'];label=f"{tokens}-{case['pattern']}-axis{case['axis']}"
        with (rope_root/label/'result.tensor').open('rb') as f:
            assert f.readline()==b'KADAN_H3_QK_ROPE_V1\n'
            assert list(map(int,f.readline().split()))==[tokens,32,3,64]
            assert f.readline()==b'F32LE\n';payload=f.read()
        assert hashlib.sha256(payload).hexdigest()==case['output_sha256']
        qkv=torch.from_numpy(np.frombuffer(payload,dtype='<f4').copy().reshape(1,tokens,32,3,64))
        attended=namespace['_sdpa_attention'](qkv[:,:,:,0,:],qkv[:,:,:,1,:],qkv[:,:,:,2,:])
        expected=F.linear(attended.reshape(tokens,2048),weights['weight'],weights['bias']).numpy()
        destination=output/label;destination.mkdir()
        (destination/'input.f32').write_bytes(payload)
        completed=subprocess.run([str(binary),str(checkpoint.parent),checkpoint.name,str(destination/'input.f32'),str(destination/'result.tensor')],check=True,capture_output=True,text=True)
        with (destination/'result.tensor').open('rb') as f:
            assert f.readline()==b'KADAN_H3_ATTENTION_V1\n'
            assert list(map(int,f.readline().split()))==[tokens,2048]
            assert f.readline()==b'F32LE\n';data=f.read()
        actual=np.frombuffer(data,dtype='<f4').reshape(tokens,2048)
        absolute=np.abs(actual-expected);bound=2e-4+2e-4*np.abs(expected)
        assert np.isfinite(actual).all() and np.all(absolute<=bound),(label,float(absolute.max()))
        cases.append(dict(tokens=tokens,pattern=case['pattern'],axis=case['axis'],values=actual.size,
            max_absolute_error=float(absolute.max()),max_tolerance_ratio=float((absolute/bound).max()),
            input_sha256=case['output_sha256'],output_sha256=hashlib.sha256(data).hexdigest(),stdout=completed.stdout.strip()))
    after=checkpoint.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    report=dict(reference='Hash-pinned H3 _sdpa_attention + Torch CPU F32 linear with real block0 to_out',
        source_sha256=PINNED,prior_rope_report_sha256=ROPE_REPORT,header_sha256=hashlib.sha256(encoded).hexdigest(),selected_tensor_sha256=hashes,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),reference_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch_version=torch.__version__,cpu_threads=1,interop_threads=1,cpu_affinity=[cpu],gpu_execution=False,
        full_video_generation=False,production_fp16_parity=False,atol=2e-4,rtol=2e-4,cases=cases,
        compared_values=sum(c['values'] for c in cases),max_absolute_error=max(c['max_absolute_error'] for c in cases),
        max_tolerance_ratio=max(c['max_tolerance_ratio'] for c in cases))
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))


if __name__=='__main__':run(*(Path(p).resolve() for p in sys.argv[1:]))
