"""Development block only: causal substitutions and bounded-K projection experiment."""
from contextlib import nullcontext
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch

from trace_binding import verify_binding
from torch.nn.attention import sdpa_kernel, SDPBackend
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock

from diagnose_block7 import manual
from metrics import compare
from precision_oracle import assess
from replay import inputs, module


def bounded_k_projection(value,weight,chunk=1024):
    # Candidate only: FP32 partial GEMMs; FP64 sum of small output matrices.
    # No duplicated rows and no FP64 matrix multiplication.
    total=torch.zeros((*value.shape[:-1],weight.shape[0]),device=value.device,dtype=torch.float64)
    for start in range(0,value.shape[-1],chunk):
        total.add_((value[...,start:start+chunk].contiguous()@weight[:,start:start+chunk].t().contiguous()).double())
    return total.to(value.dtype)


@torch.no_grad()
def main():
    binding=verify_binding("/capture","/traces")
    Path("/evidence/input-binding.json").write_text(json.dumps(binding,indent=2))
    assert torch.cuda.device_count()==1
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False
    data=torch.load('/capture/block-07.pt',weights_only=True,map_location='cpu',mmap=True)
    block=module(QwenImage21TransformerBlock,data['config'],data['state'],'cuda',torch.float32)
    args=inputs(data,'cuda',torch.float32,rows=128)
    oracle=torch.load('/oracle/oracle-output.pt',weights_only=True,map_location='cpu')
    oracle_stages=torch.load('/evidence/oracle-stages.pt',weights_only=True,map_location='cpu')
    split={'norm','qkv','attention','out','mlp'};report={};started=time.monotonic()
    for backend in ('default','math'):
        context=sdpa_kernel(SDPBackend.MATH) if backend=='math' else nullcontext()
        with context:
            saved=torch.load(Path('/traces')/f'{backend}-intermediates.pt',weights_only=True,map_location='cpu')
            corrected=torch.load(Path('/evidence')/f'{backend}-same-input-fp64.pt',weights_only=True,map_location='cpu')
            baseline,trace=manual(block,args,split)
            assert torch.equal(baseline.cpu(),saved['split']['output'])
            def verdict(value):return assess(saved['reference']['output'],value,oracle)
            result=dict(baseline=verdict(baseline),substitutions={},candidates={})
            for name,value in corrected.items():
                assert time.monotonic()-started<120
                actual,stages=manual(block,args,split,{name:value.to(device='cuda',dtype=torch.float32)})
                result['substitutions'][name]=verdict(actual)
                result['substitutions'][name]['upstream_oracle']={key:compare(stages[key],oracle_stages[key],atol=2e-5,rtol=2e-5) for key in ('residual1','mlp_out')}
            for name,owner in [('out_projection',block.attn.to_out[0]),('mlp_out',block.img_mlp.out)]:
                # Independently test both upstream attention projection and MLP output.
                for chunk in (512,1024,2048):
                    with patch.object(owner,'forward',lambda value,owner=owner,chunk=chunk:bounded_k_projection(value,owner.weight,chunk)):
                        actual,stages=manual(block,args,split)
                    result['candidates'][f'{name}-k{chunk}']=verdict(actual)
                    result['candidates'][f'{name}-k{chunk}']['upstream_oracle']={key:compare(stages[key],oracle_stages[key],atol=2e-5,rtol=2e-5) for key in ('residual1','mlp_out')}
            report[backend]=result
    Path('/evidence/correction-report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({b:{k:{n:v['virtual_vs_fp64']['violations'] for n,v in row[k].items()} for k in ('substitutions','candidates')} for b,row in report.items()}),flush=True)


if __name__=='__main__':main()
