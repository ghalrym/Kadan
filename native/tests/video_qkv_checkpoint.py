"""Explicit selected-tensor F32 CPU reference; not FP16/full H3 generation parity.

Usage: python video_qkv_checkpoint.py COMPONENT CHECKPOINT NEW_OUTPUT_DIRECTORY
Requires Torch/NumPy only for this optional real-checkpoint validation.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

# Keep optional reference dependencies out of the native build and fixture tests.
import numpy as np
import torch
import torch.nn.functional as F


def run(binary, checkpoint, output):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    output.mkdir(parents=True,exist_ok=False)
    before=checkpoint.stat()
    names={'norm1.weight':(2048,), 'attn.to_qkv.weight':(6144,2048), 'attn.to_qkv.bias':(6144,)}
    weights={};hashes={}
    with checkpoint.open('rb') as stream:
        length=struct.unpack('<Q',stream.read(8))[0]
        assert length<=1024*1024
        encoded=stream.read(length);header=json.loads(encoded)
        for name,shape in names.items():
            tensor=header['decoder.transformer_blocks.0.'+name]
            assert tensor['dtype']=='F16' and tuple(tensor['shape'])==shape
            start,end=tensor['data_offsets']
            assert end-start==int(np.prod(shape))*2 and 0<=start<end<=before.st_size-length-8
            stream.seek(8+length+start);data=stream.read(end-start)
            assert len(data)==end-start
            hashes[name]=hashlib.sha256(data).hexdigest()
            weights[name]=torch.from_numpy(np.frombuffer(data,dtype='<f2').astype(np.float32).reshape(shape))
    cases=[];rng=torch.Generator(device='cpu').manual_seed(42)
    for tokens in range(1,9):
        for pattern in ('zero','ramp','random'):
            if pattern=='zero':x=torch.zeros((tokens,2048),dtype=torch.float32)
            elif pattern=='ramp':x=((torch.arange(tokens*2048,dtype=torch.float32)%257)-128).reshape(tokens,2048)/128
            else:x=torch.randn((tokens,2048),generator=rng,dtype=torch.float32)
            normalized=F.rms_norm(x,(2048,),weights['norm1.weight'],eps=1e-5)
            expected=F.linear(normalized,weights['attn.to_qkv.weight'],weights['attn.to_qkv.bias']).reshape(tokens,32,3,64)
            assert torch.isfinite(expected).all()
            case=output/f'{tokens}-{pattern}';case.mkdir()
            (case/'input.f32').write_bytes(x.numpy().astype('<f4').tobytes())
            result=subprocess.run([str(binary),str(checkpoint.parent),checkpoint.name,str(case/'input.f32'),str(case/'qkv.tensor')],capture_output=True,text=True,check=True)
            with (case/'qkv.tensor').open('rb') as f:
                assert f.readline()==b'KADAN_H3_DECODER_QKV_V1\n'
                assert list(map(int,f.readline().split()))==[tokens,32,3,64]
                assert f.readline()==b'F32LE\n'
                payload=f.read()
            actual=np.frombuffer(payload,dtype='<f4').reshape(tokens,32,3,64)
            reference=expected.numpy()
            absolute=np.abs(actual-reference);bound=2e-4+2e-4*np.abs(reference)
            assert np.isfinite(actual).all() and np.all(absolute<=bound),(tokens,pattern,float(absolute.max()))
            cases.append(dict(tokens=tokens,pattern=pattern,values=actual.size,max_absolute_error=float(absolute.max()),
                max_tolerance_ratio=float((absolute/bound).max()),sha256=hashlib.sha256(payload).hexdigest(),stdout=result.stdout.strip()))
    after=checkpoint.stat()
    assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    report=dict(reference='Torch CPU F32 rms_norm + linear; pre-attention per-head QKV',torch_version=torch.__version__,
        cpu_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),cpu_affinity=[cpu],gpu_execution=False,
        full_video_generation=False,production_fp16_parity=False,checkpoint_bytes=before.st_size,
        header_sha256=hashlib.sha256(encoded).hexdigest(),selected_tensor_sha256=hashes,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),reference_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        atol=2e-4,rtol=2e-4,cases=cases,compared_values=sum(c['values'] for c in cases),
        max_absolute_error=max(c['max_absolute_error'] for c in cases),max_tolerance_ratio=max(c['max_tolerance_ratio'] for c in cases))
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))


if __name__=='__main__':run(*(Path(p).resolve() for p in sys.argv[1:]))
