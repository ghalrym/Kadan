"""Explicit CPU-only real Qwen3-TTS projection parity; never synthesizes audio."""
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


def main(binary,checkpoint):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    checkpoint=checkpoint.resolve(strict=True);before=checkpoint.stat();weights={};hashes={}
    prefix='talker.text_projection.'
    with checkpoint.open('rb') as stream:
        size=struct.unpack('<Q',stream.read(8))[0]
        if size>1024*1024:raise ValueError('Header limit')
        encoded=stream.read(size);header=json.loads(encoded)
        for name,row in header.items():
            if not name.startswith(prefix):continue
            assert row['dtype']=='BF16'
            start,end=row['data_offsets'];assert 0<=start<end<=before.st_size-size-8
            assert end-start<=2048*2048*2
            stream.seek(8+size+start);data=stream.read(end-start)
            assert len(data)==end-start==math.prod(row['shape'])*2
            hashes[name]=hashlib.sha256(data).hexdigest()
            weights[name.removeprefix(prefix)]=torch.frombuffer(bytearray(data),dtype=torch.bfloat16).clone().float().reshape(row['shape'])
    # Separate primitive reference, same equations as pinned ResizeMLP, CPU FP32.
    first=nn.Linear(2048,2048).eval();second=nn.Linear(2048,2048).eval()
    first.load_state_dict({k.removeprefix('linear_fc1.'):v for k,v in weights.items() if k.startswith('linear_fc1.')})
    second.load_state_dict({k.removeprefix('linear_fc2.'):v for k,v in weights.items() if k.startswith('linear_fc2.')})
    generator=torch.Generator().manual_seed(42)
    cases={'zero':torch.zeros(2048),'ramp':(torch.arange(2048)%19-9).float()/16,
           'seeded':torch.randn(2048,generator=generator)*.25};rows=[]
    with tempfile.TemporaryDirectory() as directory,torch.no_grad():
        for name,values in cases.items():
            path=Path(directory)/name;data=values.numpy().tobytes();path.write_bytes(data)
            result=json.loads(subprocess.check_output([str(binary),str(checkpoint.parent),checkpoint.name,str(path)]))
            actual=torch.tensor(result['projected']);expected=second(torch.nn.functional.silu(first(values)))
            assert actual.shape==expected.shape and torch.isfinite(actual).all()
            error=(actual-expected).abs();assert bool((error<=1e-4+expected.abs()*1e-4).all()),float(error.max())
            assert result['resident_bytes']==0 and not result['full_tts_generation'] and not result['gpu_execution']
            rows.append(dict(case=name,input_sha256=hashlib.sha256(data).hexdigest(),
                output_sha256=hashlib.sha256(actual.numpy().tobytes()).hexdigest(),values=actual.numel(),
                max_absolute_error=float(error.max()),mean_absolute_error=float(error.mean())))
    after=checkpoint.stat();assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
    print(json.dumps(dict(checkpoint=str(checkpoint),checkpoint_bytes=before.st_size,
        header_sha256=hashlib.sha256(encoded).hexdigest(),selected_tensor_sha256=hashes,
        torch_version=torch.__version__,cpu_threads=1,atol=1e-4,rtol=1e-4,cases=rows,
        gpu_execution=False,full_tts_generation=False),indent=2))


if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]))
