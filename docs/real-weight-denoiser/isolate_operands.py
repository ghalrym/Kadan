"""CPU FP64 isolated operations on identical saved FP32 operands, block 7 only."""
import json
from pathlib import Path

import torch

from trace_binding import verify_binding

from metrics import compare
from precision_oracle import normalized


@torch.no_grad()
def main():
    binding=verify_binding("/capture","/traces")
    Path("/evidence/input-binding.json").write_text(json.dumps(binding,indent=2))
    data=torch.load('/capture/block-07.pt',weights_only=True,map_location='cpu',mmap=True)
    state=data['state'];eps=data['config']['eps'];out=Path('/evidence')
    report={}
    projections={'q_projection':('mod1','attn.to_q.weight'),
        'k_projection':('mod1','attn.to_k.weight'),'v_projection':('mod1','attn.to_v.weight'),
        'out_projection':('attention','attn.to_out.0.weight'),
        'mlp_gate':('mod2','img_mlp.gate_layer.weight'),'mlp_proj':('mod2','img_mlp.proj.weight'),
        'mlp_out':('mlp_product','img_mlp.out.weight')}
    for backend in ('default','math'):
        saved=torch.load(Path('/traces')/f'{backend}-intermediates.pt',weights_only=True,map_location='cpu')
        report[backend]={};corrections={}
        for path in ('reference','split'):
            trace=saved[path];corrected={}
            for name,(source,weight) in projections.items():
                operands=trace[source].double()
                if name=='out_projection':operands=operands.flatten(2)
                corrected[name]=operands@state[weight].double().t()
            corrected['norm1']=normalized(data['hidden'][:,:128].double(),eps)
            corrected['norm2']=normalized(trace['residual1'].double(),eps)
            for name,source,weight in [('q_norm','q_projection','attn.norm_q.weight'),('k_norm','k_projection','attn.norm_k.weight')]:
                value=trace[source].double().reshape(1,128,32,128)
                corrected[name]=normalized(value,eps,True)*state[weight].double()
            gate=trace['mlp_gate'].double()
            corrected['mlp_silu']=gate*torch.sigmoid(gate)
            corrected['mlp_product']=trace['mlp_silu'].double()*trace['mlp_proj'].double()
            corrected['output']=trace['residual1'].double()+trace['gate2_tanh'].double()*trace['mlp_out'].double()
            report[backend][path]={name:compare(trace[name],value,atol=2e-5,rtol=2e-5) for name,value in corrected.items()}
            if path=='split':corrections=corrected
        torch.save(corrections,out/f'{backend}-same-input-fp64.pt')
    (out/'same-input-report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
