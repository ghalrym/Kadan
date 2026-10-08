"""Predetermined captured-block holdouts; frozen candidate, unchanged final gates."""
import argparse
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock

from controlled_projection import ControlledProjection, CANDIDATE_ID
from diagnose_block7 import manual
from precision_oracle import short_block, assess
from replay import inputs, module, eager
from trace_binding import sha256

BLOCKS=(0,15,31)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cpu-oracles',action='store_true');args_cli=parser.parse_args()
    root=Path('/capture');out=Path('/evidence')
    manifest=json.loads((root/'manifest.json').read_text())
    identity=dict(candidate=CANDIDATE_ID,candidate_source_sha256=sha256(Path(__file__).with_name('controlled_projection.py')),
        oracle_source_sha256=sha256(Path(__file__).with_name('precision_oracle.py')),
        capture_manifest_sha256=sha256(root/'manifest.json'),blocks=list(BLOCKS))
    if not args_cli.cpu_oracles:
        assert torch.cuda.device_count()==1
        torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch.backends.cuda.matmul.allow_tf32=False
    results={}
    for index in BLOCKS:
        row=manifest['blocks'][index];assert sha256(root/row['file'])==row['sha256']
        data=torch.load(root/row['file'],map_location='cpu',weights_only=True,mmap=True)
        if args_cli.cpu_oracles:
            oracle,_=short_block(data)
            torch.save(oracle,out/f'holdout-{index:02}-oracle.pt')
            results[index]=dict(capture_sha256=row['sha256'],oracle_sha256=sha256(out/f'holdout-{index:02}-oracle.pt'))
            del data,oracle
            continue
        metadata=json.loads((out/'holdout-oracles.json').read_text())
        assert metadata['identity']==identity
        assert metadata['results'][str(index)]['capture_sha256']==row['sha256']
        assert sha256(out/f'holdout-{index:02}-oracle.pt')==metadata['results'][str(index)]['oracle_sha256']
        oracle=torch.load(out/f'holdout-{index:02}-oracle.pt',weights_only=True,map_location='cpu')
        block=module(QwenImage21TransformerBlock,data['config'],data['state'],'cuda',torch.float32)
        args=inputs(data,'cuda',torch.float32,rows=128)
        owners=[block.attn.to_out[0],block.img_mlp.out];ops=[ControlledProjection(owner.weight) for owner in owners]
        results[index]={}
        for backend in ('default','math'):
            context=sdpa_kernel(SDPBackend.MATH) if backend=='math' else nullcontext()
            with context:
                original=eager(block,*args)
                reconstructed,_=manual(block,args,set())
                assert torch.equal(original,reconstructed), "Pinned eager differs from reconstruction"
                split={'norm','qkv','attention','out','mlp'}
                baseline,_=manual(block,args,split)
                result=dict(baseline=assess(original,baseline,oracle))
                with ExitStack() as stack:
                    for owner,op in zip(owners,ops):stack.enter_context(patch.object(owner,'forward',op))
                    candidate,_=manual(block,args,split)
                    result['candidate']=assess(original,candidate,oracle)
                results[index][backend]=result
        del data,block,args,owners,ops,owner,op,original,reconstructed,baseline,candidate,oracle,_
        torch.cuda.empty_cache()
    filename='holdout-oracles.json' if args_cli.cpu_oracles else 'holdout-results.json'
    (out/filename).write_text(json.dumps(dict(identity=identity,results=results),indent=2))


if __name__=='__main__':main()
