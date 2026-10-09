"""Review-only real-weight arithmetic, teacher-forced and free-running replay."""
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21TransformerBlock, QwenImage21KVLayerCache, QwenImage21AdaLayerNormContinuous,
)
from api.inference.image.sequence_parallel_attention import TokenShard, cached_block, compact_prefix, sequence_to_heads, heads_to_sequence
from capture import digest, REVISION
from metrics import compare

ROOT=Path('/capture')
REPORT=Path('/evidence')


def module(kind, args, state, device, dtype):
    with torch.device('meta'):
        result=kind(**args).to(dtype=dtype)
    result.to_empty(device=device)
    result.load_state_dict(state,strict=True)
    return result.eval()


def packet(index):
    return torch.load(ROOT/f'block-{index:02}.pt',map_location='cpu',weights_only=True,mmap=True)


def inputs(data,device,dtype,rows=None,include_hidden=True):
    hidden=data['hidden'] if rows is None else data['hidden'][:,:rows]
    prefix=tuple(v.to(device=device,dtype=dtype) for v in data['prefix'])
    mask=data['key_valid']
    if mask is not None:
        mask=mask[:,:prefix[0].shape[1]+hidden.shape[1]].to(device)
    return hidden.to(device=device,dtype=dtype) if include_hidden else None,data['modulation'].to(device=device,dtype=dtype),data['rotary'][:hidden.shape[1]].to(device),prefix,mask


def eager(block,hidden,modulation,rotary,prefix,mask):
    cache=QwenImage21KVLayerCache();cache.store(*prefix)
    return block(hidden,modulation,rotary_emb=rotary,
        target_token_mask=torch.ones(hidden.shape[1],dtype=torch.bool,device=hidden.device),
        layer_cache=cache,kv_cache_mode='cached',attention_mask=None if mask is None else mask[:,None,None])


def audit(name,actual,reference,rank,control):
    tolerance=2e-5 if name.startswith('fp32-') else .02
    row=dict(name=name,rank=rank,**compare(actual,reference,atol=tolerance,rtol=tolerance))
    with (REPORT/f'numerics-rank-{rank}.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')
    statuses=[None,None];dist.all_gather_object(statuses,row,group=control)
    if any(value['violations'] for value in statuses):
        raise AssertionError('Fixed numerical gate failed: '+name)


@torch.no_grad()
def main():
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    device=torch.device('cuda',rank)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(22*1024**3/torch.cuda.get_device_properties(rank).total_memory,rank)
    dist.init_process_group('nccl',timeout=timedelta(seconds=120),device_id=device)
    control=dist.new_group(backend='gloo',timeout=timedelta(seconds=120))
    started=time.monotonic()
    try:
        manifest=json.loads((ROOT/'manifest.json').read_text())
        assert manifest['origin']=='real-eager' and manifest['revision']==REVISION and len(manifest['blocks'])==32
        for row in [*manifest['blocks'],manifest['tail']]:
            assert digest(ROOT/row['file'])==row['sha256'], 'Capture digest changed'
        # Bit-exact exchange round-trip independently of any arithmetic tolerance.
        data=packet(0);hidden=data['hidden'].to(device);shard=TokenShard(hidden.shape[1],rank,2)
        local=shard.take(hidden).unflatten(-1,(32,128))
        for value in sequence_to_heads((local,local,local)):
            assert torch.equal(heads_to_sequence(value.contiguous()),local)
        del data,hidden,local,value
        (REPORT/f'exchange-rank-{rank}.json').write_text(json.dumps({'bit_exact':True}))
        # Trained arithmetic at short sequence only; not full-sequence FP32 attention.
        for index in range(32):
            assert time.monotonic()-started<900
            data=packet(index)
            block=module(QwenImage21TransformerBlock,data['config'],data['state'],device,torch.float32)
            assert type(block.attn.processor).__qualname__==data['processor_type']
            hidden,mod,rope,prefix,mask=inputs(data,device,torch.float32,rows=128)
            reference=eager(block,hidden,mod,rope,prefix,mask)
            shard=TokenShard(hidden.shape[1],rank,2)
            actual=cached_block(block,shard.take(hidden),mod,shard.take(rope,dim=0),
                compact_prefix(*prefix,'ulysses',rank,2),shard,control_group=control,
                mode='ulysses',key_valid=mask,request_id='real-short-fp32',step=1,block_index=index)
            # FP32 uses the previously declared tighter tolerance, not BF16 tolerance.
            audit(f'fp32-short-{index}',actual,shard.take(reference),rank,control)
            del data,block,hidden,mod,rope,prefix,mask,reference,actual
            torch.cuda.empty_cache()
        blocks=[]
        for index in range(32):
            data=packet(index)
            blocks.append(module(QwenImage21TransformerBlock,data['config'],data['state'],device,torch.bfloat16))
            assert type(blocks[-1].attn.processor).__qualname__==data['processor_type']
            del data
        timings=[]
        # Full-length teacher forcing: reset to the same captured eager input per block.
        for index,block in enumerate(blocks):
            data=packet(index);hidden,mod,rope,prefix,mask=inputs(data,device,torch.bfloat16)
            shard=TokenShard(hidden.shape[1],rank,2)
            reference=eager(block,hidden,mod,rope,prefix,mask)
            audit(f'eager-replay-{index}',reference,data['expected'],rank,control)
            actual=cached_block(block,shard.take(hidden),mod,shard.take(rope,dim=0),
                compact_prefix(*prefix,'ulysses',rank,2),shard,control_group=control,
                mode='ulysses',key_valid=mask,request_id='real-teacher',step=1,block_index=index)
            audit(f'bf16-teacher-{index}',actual,shard.take(data['expected']),rank,control)
            del data,hidden,mod,rope,prefix,mask,reference,actual
        # Stage gates: no hidden-state reset between blocks; compare each layer to capture.
        for length in (1,4,32):
            data=packet(0);first,_,_,_,_=inputs(data,device,torch.bfloat16);del data
            for mode in ('single','ulysses'):
                # Single GPU baseline is rank0 only; rank1 remains idle at control barrier.
                dist.barrier(group=control)
                torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
                hidden=first if mode=='single' else shard.take(first)
                elapsed=0.
                for index in range(length):
                    data=packet(index)
                    _,mod,rope,prefix,mask=inputs(data,device,torch.bfloat16,include_hidden=False)
                    if mode=='ulysses': prefix=compact_prefix(*prefix,mode,rank,2)
                    torch.cuda.synchronize(device);begin=time.perf_counter()
                    if mode=='single' and rank==0:
                        hidden=eager(blocks[index],hidden,mod,rope,prefix,mask)
                    elif mode=='ulysses':
                        hidden=cached_block(blocks[index],hidden,mod,shard.take(rope,dim=0),prefix,shard,
                            control_group=control,mode=mode,key_valid=mask,request_id=f'chain-{length}',step=1,block_index=index)
                    torch.cuda.synchronize(device);elapsed+=time.perf_counter()-begin
                    # Idle rank reports the captured reference as its neutral comparator.
                    actual=hidden if rank==0 or mode=='ulysses' else data['expected']
                    reference=data['expected'] if mode=='single' else shard.take(data['expected'])
                    audit(f'{mode}-chain-{length}-layer-{index}',actual,reference,rank,control)
                    del data,mod,rope,prefix,mask
                if length==32:
                    tail=torch.load(ROOT/'tail.pt',weights_only=True,map_location='cpu',mmap=True)
                    norm=module(QwenImage21AdaLayerNormContinuous,dict(embedding_dim=4096,conditioning_embedding_dim=4096,eps=tail['eps']),tail['norm_state'],device,torch.bfloat16)
                    project=module(torch.nn.Linear,dict(in_features=4096,out_features=64,bias=False),tail['projection_state'],device,torch.bfloat16)
                    if mode=='ulysses' or rank==0:
                        mask=tail['target_mask'].to(device)
                        if mode=='ulysses': mask=shard.take(mask,dim=0)
                        output=project(norm(hidden,tail['temb'].to(device),mask))
                    else: output=tail['expected']
                    reference=tail['expected'] if mode=='single' else shard.take(tail['expected'])
                    audit(f'{mode}-chain-final-projection',output,reference,rank,control)
                    del norm,project,tail,output,reference
                timings.append(dict(mode=mode,length=length,rank=rank,seconds=elapsed,
                    allocated_peak=torch.cuda.max_memory_allocated(device),reserved_peak=torch.cuda.max_memory_reserved(device)))
                del hidden
            del first
        (REPORT/f'audited-chain-rank-{rank}.json').write_text(json.dumps(timings,indent=2))
        # Separate repeated complete-chain timing from streamed numerical audits.
        # No disk reads, per-layer comparison copies or weight construction timed.
        # Both modes include final norm/projection; Ulysses includes sharding,
        # control consensus, all exchanges/layouts and one final output gather.
        begin=time.perf_counter(); contexts=[]
        for index in range(32):
            data=packet(index)
            _,mod,rope,prefix,mask=inputs(data,device,torch.bfloat16,include_hidden=False)
            contexts.append((mod,rope,prefix,mask,compact_prefix(*prefix,'ulysses',rank,2)))
            del data
        data=packet(0);first=data['hidden'].to(device);del data
        tail=torch.load(ROOT/'tail.pt',weights_only=True,map_location='cpu',mmap=True)
        norm=module(QwenImage21AdaLayerNormContinuous,dict(embedding_dim=4096,conditioning_embedding_dim=4096,eps=tail['eps']),tail['norm_state'],device,torch.bfloat16)
        project=module(torch.nn.Linear,dict(in_features=4096,out_features=64,bias=False),tail['projection_state'],device,torch.bfloat16)
        temb=tail['temb'].to(device);target=tail['target_mask'].to(device)
        torch.cuda.synchronize(device);preparation=time.perf_counter()-begin
        benchmarks=[]
        for mode in ('single','ulysses'):
            torch.cuda.reset_peak_memory_stats(device)
            times=[]
            for repeat in range(7):
                assert time.monotonic()-started<900
                dist.barrier(group=control);torch.cuda.synchronize(device)
                begin=time.perf_counter()
                hidden=first if mode=='single' else shard.take(first)
                if mode=='ulysses' or rank==0:
                    for index,(mod,rope,prefix,mask,owned) in enumerate(contexts):
                        if mode=='single': hidden=eager(blocks[index],hidden,mod,rope,prefix,mask)
                        else:
                            hidden=cached_block(blocks[index],hidden,mod,shard.take(rope,dim=0),owned,shard,
                                mode=mode,key_valid=mask,control_group=control,request_id=f'timed-{repeat}',step=1,block_index=index)
                    output=project(norm(hidden,temb,target if mode=='single' else shard.take(target,dim=0)))
                    if mode=='ulysses':
                        parts=[torch.empty_like(output) for _ in range(2)]
                        dist.all_gather(parts,output)
                        output=torch.cat(parts,dim=1)[:,:shard.total]
                    torch.cuda.synchronize(device)
                    duration=time.perf_counter()-begin
                else:
                    output=tail['expected'];duration=None
                dist.barrier(group=control)
                # Outside timing, every repetition must still pass fixed acceptance.
                audit(f'{mode}-repeat-{repeat}',output,tail['expected'],rank,control)
                if repeat>=2: times.append(duration)
                del hidden,output
                if mode=='ulysses': del parts
            benchmarks.append(dict(mode=mode,rank=rank,seconds=times,
                allocated_peak=torch.cuda.max_memory_allocated(device),reserved_peak=torch.cuda.max_memory_reserved(device)))
        (REPORT/f'timings-rank-{rank}.json').write_text(json.dumps(dict(preparation_s=preparation,benchmarks=benchmarks),indent=2))
    except BaseException as exc:
        (REPORT/f'error-rank-{rank}.json').write_text(json.dumps(dict(type=type(exc).__name__,error=str(exc)[:2000])))
        os._exit(72)
    else:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
