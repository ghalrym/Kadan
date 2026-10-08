"""Bounded two-rank communication probe. Run only after live GPU ownership release."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist

from api.inference.image.parallel import sequence_to_heads, heads_to_sequence, gather_target_kv, TokenShard, cached_block, compact_prefix
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock, QwenImage21KVLayerCache


@torch.no_grad()
def parity(rank, started, control):
    reports=[]
    torch.backends.cuda.matmul.allow_tf32=False
    for dtype in (torch.float32, torch.bfloat16):
        for dimension, heads, depth, prefix_rows, rows, masked in (
            (32,4,8,27,16,False), (32,4,8,27,17,True),
            (32,4,8,7,15,True), (4096,32,128,27,64,False)):
            assert time.monotonic()-started<120, 'Parity deadline'
            torch.manual_seed(121+rows)
            block=QwenImage21TransformerBlock(dimension,heads,depth).to(dtype=dtype,device=rank).eval()
            total=prefix_rows+rows
            joint=torch.randn(1,total,dimension,dtype=dtype,device=rank)
            modulation=torch.randn(2,4*dimension,dtype=dtype,device=rank)
            angles=torch.arange(total,device=rank).float()[:,None]*torch.linspace(.1,.9,depth//2,device=rank)[None]
            rotary=torch.polar(torch.ones_like(angles),angles)
            valid=torch.ones(1,total,dtype=torch.bool,device=rank) if masked else None
            if valid is not None: valid[:,1]=False;valid[:,-2]=False
            cache=QwenImage21KVLayerCache()
            segments=[(0,3,True),(3,7,False)] if prefix_rows==7 else [(0,prefix_rows,True)]
            block(joint,modulation,rotary_emb=rotary,target_token_mask=torch.arange(total,device=rank)>=prefix_rows,
                layer_cache=cache,kv_cache_mode='extract',cache_write_slice=slice(0,prefix_rows),segments=segments,key_valid=valid)
            hidden=torch.randn(1,rows,dimension,dtype=dtype,device=rank)
            expected=block(hidden,modulation,rotary_emb=rotary[prefix_rows:],target_token_mask=torch.ones(rows,dtype=torch.bool,device=rank),
                layer_cache=cache,kv_cache_mode='cached',attention_mask=None if valid is None else valid[:,None,None])
            shard=TokenShard(rows,rank,2)
            for mode in ('all_gather','ulysses'):
                prefix=compact_prefix(*cache.get(),mode,rank,2)
                before=tuple(value.clone() for value in prefix)
                local=cached_block(block,shard.take(hidden),modulation,shard.take(rotary[prefix_rows:],dim=0),prefix,shard,mode=mode,key_valid=valid, control_group=control, request_id=f"parity-{dtype}-{rows}-{dimension}", step=1)
                parts=[torch.empty_like(local) for _ in range(2)]
                dist.all_gather(parts,local)
                actual=torch.cat(parts,dim=1)[:,:rows]
                tolerance=2e-5 if dtype==torch.float32 else .02
                torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance)
                for original,retained in zip(before,prefix):torch.testing.assert_close(original,retained,rtol=0,atol=0)
                reports.append(dict(mode=mode,dtype=str(dtype),dimension=dimension,heads=heads,head_width=depth,
                    prefix_rows=prefix_rows,target_rows=rows,max_abs_error=float((actual.float()-expected.float()).abs().max()),
                    rmse=float(((actual.float()-expected.float())**2).mean().sqrt()),
                    prefix_storage_bytes=sum(v.untyped_storage().nbytes() for v in prefix)))
                assert torch.cuda.max_memory_reserved(rank)<=2*1024**3,'Parity allocator cap'
                del local,parts,actual,prefix,before
            del block,joint,hidden,expected,cache,modulation,rotary,angles,valid
            torch.cuda.empty_cache()
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--failure', choices=['peer_exit', 'oom'])
    args = parser.parse_args()
    if args.plan:
        print(json.dumps(dict(ranks=2, shape=[1, 8192, 32, 128], dtype='bfloat16',
            warmups=3, iterations=10, deadline_s=120, per_rank_peak_reserved_cap_bytes=2*1024**3,
            modes=['kv_all_gather', 'ulysses_qkv_and_inverse', 'tp_two_all_reduces'],
            note='No model load, no compilation; live API must release both GPUs first.')))
        return
    rank = int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE']) == 2 and torch.cuda.device_count() == 2
    torch.cuda.set_device(rank)
    free, _ = torch.cuda.mem_get_info(rank)
    assert free >= 4*1024**3, 'Require 4 GiB fresh per-device physical headroom'
    torch.cuda.set_per_process_memory_fraction(2*1024**3 / torch.cuda.get_device_properties(rank).total_memory, rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=45))
    control = dist.new_group(backend='gloo', timeout=timedelta(seconds=10))
    started = time.monotonic()
    report = dict(rank=rank, device=torch.cuda.get_device_name(rank), torch=torch.__version__,
                  peer_access=torch.cuda.can_device_access_peer(rank, 1-rank), nccl_version=torch.cuda.nccl.version(), measurements=[])
    try:
        if args.failure:
            dist.barrier()
            if rank == 1:
                Path('/evidence', f'injected-{args.failure}.json').write_text(json.dumps(dict(rank=rank, failure=args.failure)))
                if args.failure == 'peer_exit':
                    os._exit(71)
                raise torch.cuda.OutOfMemoryError('Deliberate rank-local OOM injection; no oversized allocation')
            dist.barrier()
            raise AssertionError('Peer failure was not propagated')
        report['parity']=parity(rank,started,control)
        Path('/evidence',f'parity-rank-{rank}.json').write_text(json.dumps(report['parity'],indent=2))
        for rows in (1024, 4096, 8192):
            torch.manual_seed(121)
            values = tuple(torch.randn(1, rows, 32, 128, dtype=torch.bfloat16, device=rank) for _ in range(3))
            summed = torch.zeros(1, 2*rows, 4096, dtype=torch.bfloat16, device=rank)
            def gather():
                return gather_target_kv(values[1], values[2])
            def ulysses():
                exchanged = sequence_to_heads(values)
                return heads_to_sequence(exchanged[0].contiguous())
            def tp():
                # Communication-only payload; zeros avoid overflow across repeats.
                dist.all_reduce(summed)
                dist.all_reduce(summed)
                return None
            for name, operation in [('kv_all_gather', gather), ('ulysses_qkv_and_inverse', ulysses),
                                    ('tp_two_all_reduces', tp)]:
                torch.cuda.synchronize(rank)
                dist.barrier()
                timings=[]
                torch.cuda.reset_peak_memory_stats(rank)
                for iteration in range(13):
                    assert time.monotonic()-started < 120, 'Probe deadline'
                    torch.cuda.synchronize(rank)
                    before=time.perf_counter()
                    result=operation()
                    torch.cuda.synchronize(rank)
                    duration=time.perf_counter()-before
                    if iteration >= 3:timings.append(duration)
                    del result
                    assert torch.cuda.max_memory_reserved(rank) <= 2*1024**3, 'Probe allocator cap'
                logical_payload = rows * 32 * 128 * 2 * (4 if name == 'tp_two_all_reduces' else 2)
                report['measurements'].append(dict(mode=name, local_rows=rows,
                    logical_offrank_payload_bytes=logical_payload,
                    effective_payload_gib_s=logical_payload / 2**30 / statistics.median(timings),
                    median_s=statistics.median(timings), min_s=min(timings), max_s=max(timings),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(rank),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(rank)))
            del values, summed
            torch.cuda.empty_cache()
        report['elapsed_s']=time.monotonic()-started
        Path('/evidence',f'communication-rank-{rank}.json').write_text(json.dumps(report,indent=2))
    except BaseException as exc:
        Path('/evidence',f'error-rank-{rank}.json').write_text(json.dumps(dict(type=type(exc).__name__, error=str(exc)[:2000])))
        # Do not attempt a potentially blocking NCCL destructor on a failed group.
        os._exit(72)
    else:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
