"""Streaming bounded tensors and fixed full-image comparisons."""
import hashlib
import json
from pathlib import Path

import torch

from metrics import compare
from trajectory_contracts import CRITERIA, RUN_CAP, safe
from trace_binding import sha256


def tensor_identity(value):
    if value is None:return None
    cpu=value.detach().cpu().contiguous()
    return dict(shape=list(cpu.shape),dtype=str(cpu.dtype),sha256=hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest())


def measure(actual,reference,kind):
    if actual.dtype!=reference.dtype or actual.shape!=reference.shape:raise ValueError('Shape/dtype identity mismatch')
    atol,rtol=(.02,.02) if kind=='latent' else ((CRITERIA['float_atol'],0.) if kind=='float' else (CRITERIA['pixel_max'],0.))
    row=compare(actual,reference,atol=atol,rtol=rtol)
    # Float64 subtraction prevents uint8 wraparound; chunks bound CPU allocations.
    total=0.;count=actual.numel()
    for a,r in zip(actual.reshape(-1).split(262144),reference.reshape(-1).split(262144)):
        total+=float((a.double()-r.double()).abs().sum())
    row['mae']=total/count
    row['passed']=row['violations']==0 and row['nonfinite']==0
    if kind=='float':row['passed'] &= bool(((actual>=0)&(actual<=1)).all()) and row['mae']<=CRITERIA['float_mae']
    if kind=='pixel':row['passed'] &= row['mae']<=CRITERIA['pixel_mae']
    return row


class ArtifactWriter:
    def __init__(self,root):self.root=Path(root);self.rows=[]
    def admit(self,extra):
        if sum(p.stat().st_size for p in self.root.iterdir() if p.is_file())+extra>RUN_CAP:raise RuntimeError('Per-run artifact budget exhausted')
    def record(self,name):
        p=safe(self.root,name);self.rows.append(dict(file=name,bytes=p.stat().st_size,sha256=sha256(p)))
    def tensor(self,name,value):
        p=safe(self.root,name)
        if p.exists():raise ValueError('Never overwrite trajectory tensor')
        cpu=value.detach().cpu().clone().contiguous()
        self.admit(cpu.numel()*cpu.element_size()+65536)
        torch.save(cpu,p);self.admit(0);self.record(name);return cpu
    def json(self,name,value):
        p=safe(self.root,name)
        if p.exists():raise ValueError('Never overwrite identity')
        payload=json.dumps(value,indent=2);self.admit(len(payload.encode()));p.write_text(payload);self.record(name)


def normalized_rgb(image,postprocess):
    if not bool(torch.isfinite(image).all()):raise AssertionError('Nonfinite raw VAE output before clipping')
    return postprocess(image,output_type='np')
