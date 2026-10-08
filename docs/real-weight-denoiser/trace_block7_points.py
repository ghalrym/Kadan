"""CPU-only FP64 dot checks on saved operands; never a whole-block FP64 oracle."""
import json
from pathlib import Path

import torch

ROOT=Path('/capture');OUT=Path('/evidence')


def main():
    data=torch.load(ROOT/'block-07.pt',map_location='cpu',weights_only=True,mmap=True)
    result={}
    for backend in ('default','math'):
        trace=torch.load(OUT/f'{backend}-intermediates.pt',map_location='cpu',weights_only=True)
        reference,split=trace['reference'],trace['split']
        rows=[]
        for coordinate in trace['coordinates']:
            batch,token,channel=coordinate
            entry=dict(coordinate=coordinate,paths={})
            for name,values in [('reference',reference),('split',split)]:
                gate=values['gate2_tanh'][batch,token,channel]
                product=values['mlp_product'][batch,token].double()
                weight=data['state']['img_mlp.out.weight'][channel].double()
                dot=float(torch.dot(product,weight))
                gpu=float(values['mlp_out'][batch,token,channel])
                residual=float(values['residual1'][batch,token,channel])
                output=float(values['output'][batch,token,channel])
                # Exact saved FP32 operands promoted before this isolated dot.
                # Captured BF16 weights convert exactly to FP32 then FP64.
                entry['paths'][name]=dict(gate2_tanh=float(gate),residual1=residual,mlp_out_gpu=gpu,
                    mlp_out_fp64_dot=dot,gemm_rounding_vs_saved_operand_dot=gpu-dot,
                    scaled_mlp=float(gate*values['mlp_out'][batch,token,channel]),
                    output=output,fp64_final_add_of_saved_terms=residual+float(gate)*gpu)
            rows.append(entry)
        result[backend]=rows
    (OUT/'coordinate-trace-fp64.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
