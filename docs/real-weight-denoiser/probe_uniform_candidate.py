"""Both projections use each path's current operands; fixed development protocol."""
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock

from controlled_projection import ControlledProjection, CANDIDATE_ID
from diagnose_block7 import manual
from metrics import compare
from precision_oracle import assess
from replay import inputs, module
from trace_binding import verify_binding, sha256


def timed(operation):
    for _ in range(2):operation()
    torch.cuda.synchronize();samples=[]
    for _ in range(5):
        start=time.perf_counter();operation();torch.cuda.synchronize()
        samples.append((time.perf_counter()-start)*1000)
    return dict(median_ms=statistics.median(samples),samples_ms=samples)


@torch.no_grad()
def main():
    binding=verify_binding('/capture','/traces')
    assert torch.cuda.device_count()==1
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False
    data=torch.load('/capture/block-07.pt',weights_only=True,map_location='cpu',mmap=True)
    block=module(QwenImage21TransformerBlock,data['config'],data['state'],'cuda',torch.float32)
    args=inputs(data,'cuda',torch.float32,rows=128)
    oracle=torch.load('/oracle/oracle-output.pt',weights_only=True,map_location='cpu')
    oracle_stages=torch.load('/evidence/oracle-stages.pt',weights_only=True,map_location='cpu')
    owners=[block.attn.to_out[0],block.img_mlp.out]
    torch.cuda.synchronize();start=time.perf_counter()
    candidates=[ControlledProjection(owner.weight) for owner in owners]
    torch.cuda.synchronize();pack_ms=(time.perf_counter()-start)*1000
    double_weights=[owner.weight.double() for owner in owners]
    report=dict(candidate=CANDIDATE_ID,candidate_source_sha256=sha256(Path(__file__).with_name('controlled_projection.py')),
        trace_binding=binding,pack_ms=pack_ms,packed_bytes=sum(op.packed_bytes for op in candidates),
        timings_scope='GPU0 serial virtual ranks; not distributed or end-to-end latency',backends={})
    for backend in ('default','math'):
        context=sdpa_kernel(SDPBackend.MATH) if backend=='math' else nullcontext()
        with context:
            saved=torch.load(Path('/traces')/f'{backend}-intermediates.pt',weights_only=True,map_location='cpu')
            result={}
            for mode in ('baseline','two_projection_fp64','candidate'):
                with ExitStack() as stack:
                    if mode!='baseline':
                        for index,owner in enumerate(owners):
                            op=candidates[index] if mode=='candidate' else lambda x,w=double_weights[index]:(x.double()@w.t()).float()
                            stack.enter_context(patch.object(owner,'forward',op))
                    modes={}
                    for path,split in [('eager_geometry',set()),('virtual_ranks',{'norm','qkv','attention','out','mlp'})]:
                        output,stages=manual(block,args,split)
                        if mode=='baseline':
                            assert torch.equal(output.cpu(),saved['reference' if path=='eager_geometry' else 'split']['output'])
                        row=assess(saved['reference']['output'],output,oracle)
                        row['upstream_oracle']={key:compare(stages[key],oracle_stages[key],atol=2e-5,rtol=2e-5) for key in ('residual1','mlp_out')}
                        row['complete_block_timing']=timed(lambda:manual(block,args,split))
                        modes[path]=row
                    result[mode]=modes
            report['backends'][backend]=result
            # Isolated timings use identical saved local-row operands for all modes.
            operands=[saved['split']['attention'][:,:64].flatten(2).contiguous().cuda(),saved['split']['mlp_product'][:,:64].contiguous().cuda()]
            result['projection_timings']={name:dict(baseline=timed(lambda i=i:owners[i](operands[i])),
                candidate=timed(lambda i=i:candidates[i](operands[i]))) for i,name in enumerate(('attention_output','mlp_down'))}
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    Path('/evidence/uniform-candidate-report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
