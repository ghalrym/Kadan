"""Separate full-length BF16 diagnostic; never clears the failed FP32 protocol."""
from datetime import timedelta
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock, QwenImage21AdaLayerNormContinuous

from api.inference.image.parallel import TokenShard, cached_block, compact_prefix, sequence_to_heads, heads_to_sequence
from bf16_contracts import verify_capture, verdict, require_pass, aggregate, ULYSSES_SOURCE, timing_admission
from bf16_numerics import inspect_values, expected_head_ownership, advance_pair
from replay import module, inputs, eager
from trace_binding import sha256

ROOT=Path('/capture');OUT=Path('/evidence')


def packet(index):
    return torch.load(ROOT/f'block-{index:02}.pt',weights_only=True,map_location='cpu',mmap=True)


def audit(name,actual,reference,rank,control,shard=None,rank0_only=False):
    row=None if rank0_only and rank else dict(name=name,rank=rank,**inspect_values(actual,reference))
    if row is not None:
        row['global_row_offset']=None if shard is None else shard.start
        for point in [*row['failing_points'],row['max_normalized_point']]:
            if point is not None:
                index=point['flat_index'];coordinates=[]
                for size in reversed(row['shape']):coordinates.append(index%size);index//=size
                coordinates.reverse()
                if shard is not None:coordinates[1]+=shard.start
                point['global_coordinate']=coordinates
        with (OUT/f'numerics-rank-{rank}.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        if row['violations'] or row['nonfinite']:
            # One failed boundary is retained, then the coordinated gate stops.
            torch.save(dict(actual=actual.detach().cpu(),reference=reference.detach().cpu()),OUT/f'failed-{name}-rank-{rank}.pt')
    rows=[None,None];dist.all_gather_object(rows,row,group=control)
    if rank==0:
        with (OUT/'global-numerics.jsonl').open('a') as stream:stream.write(json.dumps(dict(name=name,**aggregate(rows)))+'\n')
    require_pass(rows)


@torch.no_grad()
def main():
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.cuda.set_per_process_memory_fraction(22*1024**3/torch.cuda.get_device_properties(rank).total_memory,rank)
    # Match the existing production/benchmark TF32 setting; no new BF16 cast policy.
    torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(seconds=120),device_id=device)
    control=dist.new_group(backend='gloo',timeout=timedelta(seconds=120));started=time.monotonic()
    status=verdict('incomplete')
    def save_status():
        (OUT/f'verdict-rank-{rank}.json').write_text(json.dumps(status,indent=2))
        with (OUT/f'verdict-history-rank-{rank}.jsonl').open('a') as stream:stream.write(json.dumps(status)+'\n')
    save_status()
    try:
        assert sha256('/app/api/inference/image/parallel.py')==ULYSSES_SOURCE
        manifest=verify_capture(ROOT)
        (OUT/f'identity-rank-{rank}.json').write_text(json.dumps(dict(capture_manifest_sha256=sha256(ROOT/'manifest.json'),
            ulysses_sha256=ULYSSES_SOURCE,torch=torch.__version__,dtype='bfloat16',rank=rank,
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            bf16_reduced_precision=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
            efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),math_sdp=torch.backends.cuda.math_sdp_enabled()),indent=2))
        data=packet(0);first=data['hidden'].to(device)
        assert first.shape==(1,16384,4096) and first.dtype==torch.bfloat16
        shard=TokenShard(16384,rank,2);full=first.unflatten(-1,(32,128))
        local=full[:,rank*8192:(rank+1)*8192].contiguous()
        exchanged=sequence_to_heads((local,local,local))
        expected=expected_head_ownership(full,rank,2)
        assert all(torch.equal(value,expected) for value in exchanged), 'Independent row/head ownership failed'
        assert torch.equal(heads_to_sequence(expected),local), 'Independent inverse ownership failed'
        assert torch.equal(heads_to_sequence(exchanged[0].contiguous()),local), 'Round trip failed'
        (OUT/f'exchange-rank-{rank}.json').write_text(json.dumps(dict(independent_ownership=True,inverse_ownership=True,round_trip=True)))
        del data,full,local,exchanged,expected
        blocks=[];contexts=[]
        for index in range(32):
            data=packet(index)
            block=module(QwenImage21TransformerBlock,data['config'],data['state'],device,torch.bfloat16)
            assert type(block.attn.processor).__qualname__==data['processor_type']
            blocks.append(block)
            _,mod,rope,prefix,mask=inputs(data,device,torch.bfloat16,include_hidden=False)
            contexts.append((mod,rope,prefix,mask,compact_prefix(*prefix,'ulysses',rank,2)))
            del data
        profiled={'reference':False,'parallel':False}
        def record_profile(kind,profile):
            profiled[kind]=True
            operators=[event.key for event in profile.key_averages() if 'attention' in event.key or 'mm' in event.key]
            (OUT/f'backend-{kind}-rank-{rank}.json').write_text(json.dumps(dict(operators=operators)))
        def parallel_step(index,hidden,request):
            mod,rope,prefix,mask,owned=contexts[index]
            enabled=not profiled['parallel']
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) if enabled else nullcontext() as profile:
                result=cached_block(blocks[index],hidden,mod,shard.take(rope,dim=0),owned,shard,
                    mode='ulysses',key_valid=mask,control_group=control,request_id=request,step=1,block_index=index)
            if enabled:record_profile('parallel',profile)
            return result
        def reference_step(index,hidden):
            mod,rope,prefix,mask,_=contexts[index]
            if rank==0:
                enabled=not profiled['reference']
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) if enabled else nullcontext() as profile:
                    hidden=eager(blocks[index],hidden,mod,rope,prefix,mask)
                if enabled:record_profile('reference',profile)
            dist.broadcast(hidden,src=0)
            return hidden
        # Teacher forcing, all trained blocks at full target length.
        for index in range(32):
            data=packet(index);incoming=data['hidden'].to(device)
            reference=reference_step(index,incoming.clone())
            audit(f'eager-teacher-{index}',reference,data['expected'],rank,control,rank0_only=True)
            actual=parallel_step(index,shard.take(incoming),'bf16-teacher')
            audit(f'teacher-vs-eager-{index}',actual,shard.take(reference),rank,control,shard)
            audit(f'teacher-vs-capture-{index}',actual,shard.take(data['expected']),rank,control,shard)
            del data,incoming,reference,actual
        tail=torch.load(ROOT/'tail.pt',weights_only=True,map_location='cpu',mmap=True)
        norm=module(QwenImage21AdaLayerNormContinuous,dict(embedding_dim=4096,conditioning_embedding_dim=4096,eps=tail['eps']),tail['norm_state'],device,torch.bfloat16)
        project=module(torch.nn.Linear,dict(in_features=4096,out_features=64,bias=False),tail['projection_state'],device,torch.bfloat16)
        temb=tail['temb'].to(device);target=tail['target_mask'].to(device)
        for length in (1,4,32):
            reference=first.clone();actual=shard.take(first)
            for index in range(length):
                reference,actual=advance_pair(reference,actual,lambda value:reference_step(index,value),
                    lambda value:parallel_step(index,value,f'bf16-chain-{length}'))
                data=packet(index)
                audit(f'eager-chain-{length}-{index}',reference,data['expected'],rank,control,rank0_only=True)
                audit(f'chain-{length}-{index}',actual,shard.take(reference),rank,control,shard)
                del data
            if length==32:
                reference_output=project(norm(reference,temb,target)) if rank==0 else torch.empty_like(tail['expected'],device=device)
                dist.broadcast(reference_output,src=0)
                actual_output=project(norm(actual,temb,shard.take(target,dim=0)))
                audit('eager-final',reference_output,tail['expected'],rank,control,rank0_only=True)
                audit('chain-final',actual_output,shard.take(reference_output),rank,control,shard)
                del actual_output,reference_output
            del reference,actual
        status=verdict('passed','pending');save_status()
        # Optional cached-core timings never weaken numerical acceptance.
        available=int(os.environ['KADAN_STAGE_SECONDS'])-(time.monotonic()-started)
        budgets=[None,None]
        dist.all_gather_object(budgets,available,group=control)
        decision=timing_admission(budgets)
        (OUT/f'timing-admission-rank-{rank}.json').write_text(json.dumps(decision,indent=2))
        if not decision['admitted']:
            status=verdict('passed','skipped-insufficient-budget');save_status()
        else:
            benchmarks=[]
            for mode in ('single','ulysses'):
                times=[]
                for repeat in range(7):
                    dist.barrier(group=control);torch.cuda.synchronize();begin=time.perf_counter()
                    if mode=='ulysses':
                        hidden=shard.take(first)
                        for index in range(32):hidden=parallel_step(index,hidden,f'bf16-timed-{repeat}')
                        output=project(norm(hidden,temb,shard.take(target,dim=0)))
                        parts=[torch.empty_like(output) for _ in range(2)];dist.all_gather(parts,output)
                        output=torch.cat(parts,dim=1)[:,:shard.total]
                    elif rank==0:
                        hidden=first
                        for index in range(32):
                            mod,rope,prefix,mask,_=contexts[index]
                            hidden=eager(blocks[index],hidden,mod,rope,prefix,mask)
                        output=project(norm(hidden,temb,target))
                    else:output=None
                    torch.cuda.synchronize();duration=time.perf_counter()-begin
                    dist.barrier(group=control)
                    # Full gathered output is checked once; rank1 still joins consensus.
                    audit(f'{mode}-repeat-{repeat}',output,tail['expected'],rank,control,rank0_only=True)
                    if repeat>=2:times.append(duration if rank==0 or mode=='ulysses' else None)
                benchmarks.append(dict(mode=mode,seconds=times))
            (OUT/f'timings-rank-{rank}.json').write_text(json.dumps(dict(scope='first-cached-step BF16 core only',benchmarks=benchmarks),indent=2))
            status=verdict('passed','passed-cached-core-only');save_status()
    except BaseException as exc:
        status=verdict('failed' if isinstance(exc,AssertionError) else 'incomplete');save_status()
        (OUT/f'error-rank-{rank}.json').write_text(json.dumps(dict(type=type(exc).__name__,error=str(exc)[:2000])))
        os._exit(72)
    else:dist.destroy_process_group()


if __name__=='__main__':main()
