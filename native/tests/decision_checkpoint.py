"""Explicit CPU-only real-checkpoint parity; never runs the text encoder/head transformers.

Usage: python decision_checkpoint.py NATIVE_COMPONENT CHECKPOINT
Requires CPU Torch for an independent LayerNorm/Linear/erf-GELU reference.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import torch
from torch import nn


def main(binary, checkpoint):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoint=checkpoint.resolve(strict=True)
    before=checkpoint.stat();weights={};hashes={}
    with checkpoint.open('rb') as stream:
        size=struct.unpack('<Q',stream.read(8))[0]
        if size>1024*1024:raise ValueError('Header limit')
        encoded=stream.read(size);header=json.loads(encoded)
        for name,row in header.items():
            if not name.startswith(('scorer.','act_head.')):continue
            assert row['dtype']=='F16'
            start,end=row['data_offsets'];assert 0<=start<end<=before.st_size-size-8
            assert end-start<=2*1024*1024
            stream.seek(8+size+start);data=stream.read(end-start)
            assert len(data)==end-start==math.prod(row['shape'])*2
            hashes[name]=hashlib.sha256(data).hexdigest()
            weights[name]=torch.frombuffer(bytearray(data),dtype=torch.float16).clone().float().reshape(row['shape'])
    # Matches pinned laya.common.DecisionModel terminal modules, CPU FP32 eval.
    scorer=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,1024),nn.GELU(),nn.Linear(1024,1)).eval()
    action=nn.Sequential(nn.Linear(1028,256),nn.GELU(),nn.Linear(256,2)).eval()
    scorer.load_state_dict({k.removeprefix('scorer.'):v for k,v in weights.items() if k.startswith('scorer.')})
    action.load_state_dict({k.removeprefix('act_head.'):v for k,v in weights.items() if k.startswith('act_head.')})
    generator=torch.Generator().manual_seed(42)
    cases={'zero':torch.zeros(2052),'ramp':(torch.arange(2052)%19-9).float()/16,
           'seeded':torch.randn(2052,generator=generator)*.25}
    rows=[]
    with tempfile.TemporaryDirectory() as directory,torch.no_grad():
        for name,values in cases.items():
            path=Path(directory)/name;data=values.numpy().tobytes();path.write_bytes(data)
            actual=json.loads(subprocess.check_output([str(binary),str(checkpoint.parent),checkpoint.name,str(path)]))
            expected=torch.cat((scorer(values[:1024]),action(values[1024:]))).tolist()
            output=[actual['score'],*actual['actions']]
            errors=[abs(a-b) for a,b in zip(output,expected)]
            assert all(math.isfinite(x) for x in output)
            assert all(error<=1e-4+1e-4*abs(reference) for error,reference in zip(errors,expected)),(name,output,expected)
            assert actual['resident_bytes_at_publication']==(2052+3)*4
            assert actual['resident_bytes']==0 and actual['full_decision_inference'] is False and actual['gpu_execution'] is False
            rows.append(dict(case=name,input_sha256=hashlib.sha256(data).hexdigest(),actual=output,reference=expected,max_absolute_error=max(errors)))
    after=checkpoint.stat();assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
    print(json.dumps(dict(checkpoint=str(checkpoint),checkpoint_bytes=before.st_size,
        header_sha256=hashlib.sha256(encoded).hexdigest(),selected_tensor_sha256=hashes,
        torch_version=torch.__version__,cpu_threads=1,atol=1e-4,rtol=1e-4,cases=rows,
        gpu_execution=False,full_decision_inference=False),indent=2))


if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]))
