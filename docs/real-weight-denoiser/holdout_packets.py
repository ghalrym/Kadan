"""Bounded context-only writes; immutable original weight packets are never copied."""
import json
from pathlib import Path
import uuid

import torch

from bf16_contracts import LIMIT
from holdout_contracts import BLOCK_KEYS, TAIL_KEYS, RESERVE, safe_file
from trace_binding import sha256


def tensor_bytes(value):
    if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value,(list,tuple)):return sum(tensor_bytes(item) for item in value)
    return 0


class ContextWriter:
    def __init__(self,root,base_bytes,limit=LIMIT,reserve=RESERVE):
        self.root=Path(root);self.base_bytes=base_bytes;self.limit=limit;self.reserve=reserve
        if list(self.root.iterdir()):raise ValueError('Fresh empty context directory required')
        self.run_id=uuid.uuid4().hex;self.files=[];self._ownership()

    def _ownership(self):
        (self.root/'ownership.json').write_text(json.dumps(dict(run_id=self.run_id,files=self.files)))

    def write(self,name,value,tail=False):
        if set(value)!=(TAIL_KEYS if tail else BLOCK_KEYS):raise ValueError('Only context fields may be written; weights forbidden')
        path=safe_file(self.root,name)
        if path.exists():raise ValueError('Never overwrite context')
        used=sum(p.stat().st_size for p in self.root.iterdir())
        estimate=tensor_bytes(value)+16*1024**2
        if self.base_bytes+used+estimate+self.reserve>self.limit:raise RuntimeError('32-GiB shared artifact budget exhausted before write')
        partial=safe_file(self.root,name+'.partial')
        if partial.exists():raise ValueError('Partial artifact already exists')
        torch.save(value,partial)
        if self.base_bytes+sum(p.stat().st_size for p in self.root.iterdir())+self.reserve>self.limit:
            raise RuntimeError('Actual serialized size exceeds budget; preserve partial and stop')
        partial.rename(path);self.files.append(name);self._ownership()
        return dict(file=name,sha256=sha256(path),bytes=path.stat().st_size,base_packet_sha256=value['base_packet_sha256'])


def estimate_context_bytes(root,manifest):
    root=Path(root);payload=0
    for row in manifest['blocks']:
        data=torch.load(root/row['file'],weights_only=True,map_location='cpu',mmap=True)
        payload+=tensor_bytes({key:data[key] for key in BLOCK_KEYS if key not in ('step_index','base_packet_sha256')})
    data=torch.load(root/manifest['tail']['file'],weights_only=True,map_location='cpu',mmap=True)
    payload+=tensor_bytes({key:data[key] for key in TAIL_KEYS if key not in ('step_index','base_packet_sha256')})
    base_bytes=sum(p.stat().st_size for p in root.iterdir())
    return dict(base_bytes=base_bytes,context_tensor_bytes=payload,serialization_margin_bytes=33*16*1024**2,
        evidence_reserve_bytes=RESERVE,projected_active_bytes=base_bytes+payload+33*16*1024**2+RESERVE,limit_bytes=LIMIT)
