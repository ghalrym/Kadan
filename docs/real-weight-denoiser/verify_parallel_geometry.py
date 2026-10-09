"""Separate two-rank diagnostic entrypoint; existing replay gate is unchanged.

Run only under the reviewed two-GPU supervisor/lock/resource envelope. This file
never stops services and is not invoked by launch.py. No automatic BF16 promotion.
"""
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock

from api.inference.image.sequence_parallel_attention import TokenShard, cached_block
from capture import digest
from diagnose_block7 import manual
from metrics import compare
from replay import eager, inputs, module


@torch.no_grad()
def main():
    rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(rank).total_memory,rank)
    torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(seconds=120),device_id=torch.device('cuda',rank))
    control=dist.new_group(backend='gloo',timeout=timedelta(seconds=120))
    try:
        assert dist.get_world_size()==2
        root=Path('/capture');out=Path('/evidence')
        row=json.loads((root/'manifest.json').read_text())['blocks'][7]
        assert digest(root/row['file'])==row['sha256']
        data=torch.load(root/row['file'],map_location='cpu',weights_only=True,mmap=True)
        block=module(QwenImage21TransformerBlock,data['config'],data['state'],f'cuda:{rank}',torch.float32)
        args=inputs(data,f'cuda:{rank}',torch.float32,rows=128)
        hidden,mod,rope,prefix,mask=args
        # Virtual reference uses explicit concat/head slicing and no collectives.
        virtual,_=manual(block,args,{'norm','qkv','attention','out','mlp'})
        original=eager(block,*args)
        # Independent explicit ownership for this fixed even short-block case.
        local=hidden[:,rank*64:(rank+1)*64].contiguous()
        local_rope=rope[rank*64:(rank+1)*64].contiguous()
        local_prefix=tuple(value[:,:,rank*16:(rank+1)*16].clone().contiguous() for value in prefix)
        actual=cached_block(block,local,mod,local_rope,local_prefix,TokenShard(128,rank,2),
            mode='ulysses',key_valid=mask,control_group=control,request_id='geometry-proof',step=1,block_index=7)
        expected=virtual[:,rank*64:(rank+1)*64]
        report=dict(rank=rank,capture_sha256=row['sha256'],
            distributed_vs_virtual=compare(actual,expected,atol=0,rtol=0),
            original_eager_compatibility=compare(actual,original[:,rank*64:(rank+1)*64],atol=2e-5,rtol=2e-5),
            scope='transport/layout only; independent FP64 and BF16 trajectory gates required')
        # Persist before any assertion; independent oracle can assess these later.
        torch.save(actual.cpu(),out/f'distributed-rank-{rank}.pt')
        (out/f'geometry-rank-{rank}.json').write_text(json.dumps(report,indent=2))
        statuses=[None,None];dist.all_gather_object(statuses,report,group=control)
        assert all(item['distributed_vs_virtual']['violations']==0 for item in statuses)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':main()
