"""Both paths' own saved MLP operands: one rounded down-projection substitution."""
import json
from pathlib import Path

import torch

from trace_binding import verify_binding

from precision_oracle import short_block, assess


@torch.no_grad()
def main():
    binding=verify_binding("/capture","/traces")
    Path("/evidence/input-binding.json").write_text(json.dumps(binding,indent=2))
    data=torch.load('/capture/block-07.pt',weights_only=True,map_location='cpu',mmap=True)
    oracle,trace=short_block(data)
    torch.save(trace,'/evidence/oracle-stages.pt')
    weight=data['state']['img_mlp.out.weight'].double();report={}
    for backend in ('default','math'):
        saved=torch.load(Path('/traces')/f'{backend}-intermediates.pt',weights_only=True,map_location='cpu')
        result={}
        for path in ('reference','split'):
            values=saved[path]
            down=(values['mlp_product'].double()@weight.t()).float()
            output=values['residual1']+values['gate2_tanh']*down
            result[path]=assess(saved['reference']['output'],output,oracle)
        report[backend]=result
    Path('/evidence/narrow-readout.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
