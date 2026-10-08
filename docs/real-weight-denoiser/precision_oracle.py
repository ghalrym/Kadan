"""Independent CPU FP64 short-block equations; no production/Diffusers operators."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from metrics import compare


def rotate_pairs(value, frequencies):
    """FP64 real pair rotation using exact stored complex64 coefficients."""
    pairs=value.reshape(*value.shape[:-1],-1,2)
    real,imag=pairs[...,0],pairs[...,1]
    cosine=frequencies.real.double()[None,:,None,:]
    sine=frequencies.imag.double()[None,:,None,:]
    return torch.stack((real*cosine-imag*sine,real*sine+imag*cosine),dim=-1).flatten(-2)


def normalized(value,epsilon,rms=False):
    centered=value if rms else value-value.mean(dim=-1,keepdim=True)
    return centered/(centered.square().mean(dim=-1,keepdim=True)+epsilon).sqrt()


def attention64(query,key,value,valid=None):
    # Bound score storage to one head; no SDPA or production attention helpers.
    outputs=[]
    for head in range(query.shape[2]):
        logits=(query[:,:,head]@key[:,:,head].transpose(-1,-2))/query.shape[-1]**0.5
        if valid is not None:
            if not bool(valid.any(dim=-1).all()):
                raise ValueError('Oracle requires at least one valid key per sample')
            logits=logits.masked_fill(~valid[:,None,:],-torch.inf)
        shifted=logits-logits.amax(dim=-1,keepdim=True)
        probabilities=shifted.exp()
        probabilities=probabilities/probabilities.sum(dim=-1,keepdim=True)
        outputs.append(probabilities@value[:,:,head])
    return torch.stack(outputs,dim=2)


@torch.no_grad()
def short_block(data,rows=128):
    """Mathematical FP64 reference for captured, already-quantized inputs/weights.

    Prefix and RoPE coefficients are captured constants, not regenerated FP64
    prefill. No implicit float32 conversion is permitted within these equations.
    """
    state=data['state'];config=data['config'];eps=config['eps']
    hidden=data['hidden'][:,:rows].double()
    mod=data['modulation'][:-1].double()[:,None,:]
    scale1,gate1,scale2,gate2=mod.chunk(4,dim=-1)
    heads=config['num_attention_heads'];depth=config['attention_head_dim']
    trace={}
    def keep(name,value):
        trace[name]=value
        return value
    def linear(value,name):
        return value@state[name+'.weight'].double().t()
    value=keep('mod1',normalized(hidden,eps)*(1+scale1))
    query=linear(value,'attn.to_q').reshape(*value.shape[:2],heads,depth)
    key=linear(value,'attn.to_k').reshape_as(query)
    val=linear(value,'attn.to_v').reshape_as(query)
    query=normalized(query,eps,True)*state['attn.norm_q.weight'].double()
    key=normalized(key,eps,True)*state['attn.norm_k.weight'].double()
    query=keep('q_rope',rotate_pairs(query,data['rotary'][:rows]))
    key=rotate_pairs(key,data['rotary'][:rows])
    prefix_key,prefix_value=data['prefix']
    key=torch.cat((prefix_key.double(),key),dim=1)
    val=torch.cat((prefix_value.double(),val),dim=1)
    mask=data['key_valid']
    if mask is not None:mask=mask[:,:key.shape[1]]
    result=keep('attention',attention64(query,key,val,mask))
    result=linear(result.flatten(2),'attn.to_out.0')
    hidden=keep('residual1',hidden+gate1.tanh()*result)
    value=keep('mod2',normalized(hidden,eps)*(1+scale2))
    gate=linear(value,'img_mlp.gate_layer')
    product=keep('mlp_product',(gate*torch.sigmoid(gate))*linear(value,'img_mlp.proj'))
    result=keep('mlp_out',linear(product,'img_mlp.out'))
    return keep('output',hidden+gate2.tanh()*result),trace


def assess(eager,virtual,oracle,distributed=None):
    """Independent verdicts; no result can overwrite the original comparison."""
    def numerical(a,b):
        a=a.detach().cpu().double();b=b.detach().cpu().double()
        result=compare(a,b,atol=2e-5,rtol=2e-5)
        bound=2e-5+2e-5*b.abs();error=(a-b).abs()
        coordinates=(error>bound).nonzero()
        result['failing_coordinates']=[dict(coordinate=index.tolist(),actual=float(a[tuple(index)]),
            reference=float(b[tuple(index)]),error=float(error[tuple(index)]),bound=float(bound[tuple(index)]))
            for index in coordinates[:100]]
        result['coordinates_truncated']=len(coordinates)>100
        return result
    return dict(original_eager_compatibility=numerical(virtual,eager),
        eager_vs_fp64=numerical(eager,oracle),virtual_vs_fp64=numerical(virtual,oracle),
        distributed_vs_virtual=None if distributed is None else compare(distributed,virtual,atol=0,rtol=0),
        scope='short captured block only; not BF16 trajectory acceptance')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--traces',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest=json.loads((args.capture/'manifest.json').read_text())
    row=manifest['blocks'][7];path=args.capture/row['file']
    hasher=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):hasher.update(chunk)
    assert hasher.hexdigest()==row['sha256']
    data=torch.load(path,weights_only=True,map_location='cpu',mmap=True)
    output,trace=short_block(data)
    report=dict(capture_sha256=row['sha256'],torch=torch.__version__,device='cpu',dtype='float64',backends={})
    for backend in ('default','math'):
        saved=torch.load(args.traces/f'{backend}-intermediates.pt',weights_only=True,map_location='cpu')
        result=assess(saved['reference']['output'],saved['split']['output'],output)
        result['stages']={name:dict(eager=compare(saved['reference'][name],value,atol=2e-5,rtol=2e-5),
            virtual=compare(saved['split'][name],value,atol=2e-5,rtol=2e-5)) for name,value in trace.items()}
        report['backends'][backend]=result
    torch.save(output,args.output/'oracle-output.pt')
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({key:{name:value['violations'] for name,value in result.items() if isinstance(value,dict) and 'violations' in value} for key,result in report['backends'].items()}),flush=True)


if __name__=='__main__':main()
