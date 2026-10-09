"""Explicit bounded CPU-only comparison to pinned openai-whisper, no models/GPU."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
import whisper.audio as reference

ASSET_SHA='7450ae70723a5ef9d341e3cee628c7cb0177f36ce42c44b7ed2bf3325f0f6d4c'
FILTER_SHA={80:'4f2701b1d287d74a0dc9871026e9519d98cb76426615f2539b0d151a0ae4ec2e',128:'2a5f9822897750e047c85dea37cc268d3be0ecfd23c28a5f10da129d99d05afe'}
ATOL=RTOL=2e-4

def main(binary):
    assert importlib.metadata.version('openai-whisper')=='20250625'
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    asset=Path(reference.__file__).parent/'assets/mel_filters.npz'
    assert hashlib.sha256(asset.read_bytes()).hexdigest()==ASSET_SHA
    rows=[];rng=np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as folder,torch.no_grad():
        root=Path(folder)
        for bins in (80,128):
            bank=reference.mel_filters('cpu',bins).numpy().astype('<f4').tobytes()
            assert hashlib.sha256(bank).hexdigest()==FILTER_SHA[bins]
            (root/'bank').write_bytes(bank)
            for n in (201,319,320,15999,16000):
                impulse=np.zeros(n,dtype=np.float32);impulse[0]=1;impulse[-1]=-.5
                cases={'silence':np.zeros(n,dtype=np.float32),'boundary_impulses':impulse,
                    'sine':(.25*np.sin(2*np.pi*440*np.arange(n)/16000)).astype(np.float32),
                    'seeded':(rng.standard_normal(n)*.1).astype(np.float32)}
                for name,pcm in cases.items():
                    raw=pcm.astype('<f4').tobytes();(root/'pcm').write_bytes(raw)
                    result=json.loads(subprocess.check_output([str(binary),str(bins),str(root/'bank'),str(root/'pcm')],timeout=15))
                    actual=np.asarray(result['mel'],dtype=np.float32).reshape(bins,n//160)
                    expected=reference.log_mel_spectrogram(torch.from_numpy(pcm),n_mels=bins).numpy()
                    assert actual.shape==expected.shape and np.isfinite(actual).all()
                    error=np.abs(actual-expected);bound=ATOL+RTOL*np.abs(expected)
                    assert np.all(error<=bound),(bins,n,name,float(error.max()),float((error/bound).max()))
                    assert result['resident_bytes']==0 and result['resident_bytes_at_publication']==len(bank)+len(raw)+actual.nbytes
                    assert not result['full_transcription'] and not result['gpu_execution']
                    rows.append(dict(bins=bins,samples=n,case=name,values=actual.size,input_sha256=hashlib.sha256(raw).hexdigest(),
                        output_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),max_absolute_error=float(error.max()),max_tolerance_ratio=float((error/bound).max())))
    print(json.dumps(dict(reference='openai-whisper==20250625',asset_sha256=ASSET_SHA,filter_sha256=FILTER_SHA,
        torch_version=torch.__version__,reference_source_sha256=hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),cpu_threads=1,gpu_execution=False,full_transcription=False,atol=ATOL,rtol=RTOL,cases=rows),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
