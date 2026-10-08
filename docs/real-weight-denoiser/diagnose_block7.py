"""Bounded GPU0-only arithmetic isolation; no NCCL and no API interaction."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import diffusers.models.transformers.transformer_qwenimage21 as pinned

from capture import digest
from metrics import compare
from replay import module, inputs, eager

ROOT=Path('/capture')
OUT=Path('/evidence')


def rowwise(operation,value,split):
    if split == 'padded':
        results=[]
        for rank,part in enumerate(value.chunk(2,dim=1)):
            padded=torch.zeros_like(value)
            padded[:,rank*part.shape[1]:(rank+1)*part.shape[1]]=part
            results.append(operation(padded)[:,rank*part.shape[1]:(rank+1)*part.shape[1]])
        return torch.cat(results,dim=1)
    return torch.cat([operation(part.contiguous()) for part in value.chunk(2,dim=1)],dim=1) if split else operation(value)


def details(actual,reference):
    a=actual.detach().cpu().double();r=reference.detach().cpu().double()
    error=(a-r).abs();bound=2e-5+2e-5*r.abs();ratio=error/bound
    failing=(error>bound).nonzero()
    def point(index):
        index=tuple(int(v) for v in index)
        return dict(coordinate=list(index),actual=float(a[index]),reference=float(r[index]),
            error=float(error[index]),bound=float(bound[index]),normalized_ratio=float(ratio[index]))
    worst=torch.unravel_index(ratio.argmax(),ratio.shape)
    return dict(**compare(actual,reference,atol=2e-5,rtol=2e-5),
        failing_coordinates=[point(index) for index in failing[:100]],
        coordinates_truncated=len(failing)>100,max_normalized_coordinate=point(worst))


def traced_eager(block,args):
    trace={};hooks=[]
    names={'norm1':block.img_norm1,'q_projection':block.attn.to_q,'k_projection':block.attn.to_k,
        'v_projection':block.attn.to_v,'q_norm':block.attn.norm_q,'k_norm':block.attn.norm_k,
        'out_projection':block.attn.to_out[0],'norm2':block.img_norm2,'mlp_gate':block.img_mlp.gate_layer,
        'mlp_proj':block.img_mlp.proj,'mlp_silu':block.img_mlp.activation_fn,'mlp_out':block.img_mlp.out}
    for name,owner in names.items():
        hooks.append(owner.register_forward_hook(lambda mod,ins,out,name=name: trace.__setitem__(name,out.detach().clone())))
    hooks.append(block.attn.to_q.register_forward_pre_hook(lambda mod,ins: trace.__setitem__('mod1',ins[0].detach().clone())))
    hooks.append(block.img_norm2.register_forward_pre_hook(lambda mod,ins: trace.__setitem__('residual1',ins[0].detach().clone())))
    hooks.append(block.img_mlp.proj.register_forward_pre_hook(lambda mod,ins: trace.__setitem__('mod2',ins[0].detach().clone())))
    hooks.append(block.img_mlp.out.register_forward_pre_hook(lambda mod,ins: trace.__setitem__('mlp_product',ins[0].detach().clone())))
    original=pinned.dispatch_attention_fn
    def attention(q,k,v,**kwargs):
        trace.update(q_rope=q.detach().clone(),k_prefix=k.detach().clone(),v_prefix=v.detach().clone())
        result=original(q,k,v,**kwargs);trace['attention']=result.detach().clone()
        return result
    try:
        with patch.object(pinned,'dispatch_attention_fn',attention):
            result=eager(block,*args)
        trace['output']=result.detach().clone()
        target=torch.ones(args[0].shape[1],dtype=torch.bool,device=args[0].device)
        trace['gate2_tanh']=block._modulate(trace['norm2'],args[1].chunk(2,dim=-1)[1],target)[1].tanh()
        return result,trace
    finally:
        for hook in hooks:hook.remove()


def manual(block,args,split,overrides=None):
    hidden,mod,rope,prefix,mask=args;trace={};overrides=overrides or {}
    def keep(name,value):
        value=overrides.get(name,value);trace[name]=value
        return value
    def split_linear(name):
        return 'padded' if 'padded' in split else name in split
    a=block.attn;target=torch.ones(hidden.shape[1],device=hidden.device,dtype=torch.bool)
    first,second=mod.chunk(2,dim=-1)
    normal=keep('norm1',rowwise(block.img_norm1,hidden,'norm' in split))
    normal,gate=block._modulate(normal,first,target);normal=keep('mod1',normal)
    q=keep('q_projection',rowwise(a.to_q,normal,split_linear('qkv'))).unflatten(-1,(32,128))
    k=keep('k_projection',rowwise(a.to_k,normal,split_linear('qkv'))).unflatten(-1,(32,128))
    v=keep('v_projection',rowwise(a.to_v,normal,split_linear('qkv'))).unflatten(-1,(32,128))
    q=keep('q_norm',rowwise(a.norm_q,q,'norm' in split));k=keep('k_norm',rowwise(a.norm_k,k,'norm' in split))
    q=keep('q_rope',pinned.apply_rotary_emb_qwen(q,rope,use_real=False))
    k=pinned.apply_rotary_emb_qwen(k,rope,use_real=False)
    k=keep('k_prefix',torch.cat((prefix[0],k),dim=1));v=keep('v_prefix',torch.cat((prefix[1],v),dim=1))
    def attention(q,k,v):
        return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
            attn_mask=None if mask is None else mask[:,None,None],dropout_p=0.,is_causal=False).transpose(1,2)
    if 'attention' in split:
        # Emulate fused exchange buffers/head ownership without transport.
        result=torch.cat([attention(q[:,:,start:start+16].contiguous(),k[:,:,start:start+16].contiguous(),v[:,:,start:start+16].contiguous()) for start in (0,16)],dim=2)
    else:result=attention(q,k,v)
    result=keep('attention',result)
    result=keep('out_projection',rowwise(a.to_out[0],result.flatten(2,3).contiguous(),split_linear('out')))
    hidden=keep('residual1',hidden+gate.tanh()*result)
    normal=keep('norm2',rowwise(block.img_norm2,hidden,'norm' in split))
    normal,gate=block._modulate(normal,second,target);normal=keep('mod2',normal)
    gated=keep('mlp_gate',rowwise(block.img_mlp.gate_layer,normal,split_linear('mlp')))
    projected=keep('mlp_proj',rowwise(block.img_mlp.proj,normal,split_linear('mlp')))
    activated=keep('mlp_silu',block.img_mlp.activation_fn(gated))
    product=keep('mlp_product',activated*projected)
    result=keep('mlp_out',rowwise(block.img_mlp.out,product,split_linear('mlp')))
    gate=keep('gate2_tanh',gate.tanh())
    return keep('output',hidden+gate*result),trace


def validate_controls(report):
    """Pinned-capture regression: retain failures while proving the shape control."""
    variants=report['variants']
    for name in ('full','norm_only','attention_only','padded_shape_control'):
        assert variants[name]['output']['max_abs_error']==0, name
    assert all(row['max_abs_error']==0 for row in variants['padded_shape_control']['stages'].values())
    failed=variants['all_split']['output']
    expected=[[0,39,714],[0,39,2552],[0,39,3323]] if report['backend']=='default' else [[0,39,2552],[0,39,3013]]
    assert failed['violations']==len(expected)
    assert [row['coordinate'] for row in failed['failing_coordinates']]==expected
    assert failed['max_normalized_tolerance_ratio']>1


@torch.no_grad()
def main():
    assert torch.cuda.device_count()==1
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False
    started=time.monotonic()
    manifest=json.loads((ROOT/'manifest.json').read_text())
    row=manifest['blocks'][7];assert digest(ROOT/row['file'])==row['sha256']
    data=torch.load(ROOT/row['file'],map_location='cpu',weights_only=True,mmap=True)
    block=module(pinned.QwenImage21TransformerBlock,data['config'],data['state'],'cuda:0',torch.float32)
    args=inputs(data,'cuda:0',torch.float32,rows=128)
    settings=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        math_sdp_reduced_precision=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(),mem_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
        math_sdp=torch.backends.cuda.math_sdp_enabled(),cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled(),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    (OUT/'settings.json').write_text(json.dumps(settings,indent=2))
    variants={'full':set(),'norm_only':{'norm'},'qkv_only':{'qkv'},'attention_only':{'attention'},
        'out_only':{'out'},'mlp_only':{'mlp'},'all_split':{'norm','qkv','attention','out','mlp'},
        'padded_shape_control':{'norm','attention','padded'}}
    for backend in ('default','math'):
        context=sdpa_kernel(SDPBackend.MATH) if backend=='math' else nullcontext()
        with context:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                reference,reftrace=traced_eager(block,args)
            operators=[e.key for e in profile.key_averages() if 'attention' in e.key or 'mm' in e.key]
            summary=dict(backend=backend,operators=operators,variants={})
            for name,split in variants.items():
                assert time.monotonic()-started<120
                output,trace=manual(block,args,split)
                summary['variants'][name]=dict(output=details(output,reference),
                    stages={key:compare(value,reftrace[key],atol=2e-5,rtol=2e-5) for key,value in trace.items()})
                if name=='all_split':
                    points=details(output,reference)['failing_coordinates']
                    coordinates=[p['coordinate'] for p in points]
                    torch.save(dict(reference={k:v.cpu() for k,v in reftrace.items()},
                        split={k:v.cpu() for k,v in trace.items()},coordinates=coordinates),OUT/f'{backend}-intermediates.pt')
                    # Controlled substitutions expose which upstream drift feeds final errors.
                    summary['substitutions']={}
                    for stage in ('q_projection','k_projection','v_projection','attention','out_projection','residual1','mod2','mlp_product','mlp_out'):
                        fixed,_=manual(block,args,split,{stage:reftrace[stage]})
                        summary['substitutions'][stage]=details(fixed,reference)
                del output,trace
            (OUT/f'{backend}-report.json').write_text(json.dumps(summary,indent=2))
            validate_controls(summary)
    (OUT/'complete.json').write_text(json.dumps(dict(elapsed_s=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())))


if __name__=='__main__':main()
