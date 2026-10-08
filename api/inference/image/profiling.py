"""Opt-in acceptance instrumentation; never installed by the production adapter.

Wrap after Accelerate installs offload hooks. Stage synchronization is deliberate
and its overhead must be disclosed. Only the first transformer call is optionally
traced, bounding profiler lifetime; all other phases use synchronized wall clocks.
"""
from contextlib import contextmanager, nullcontext
from functools import wraps
import time


@contextmanager
def profile_pipeline(pipeline, torch, device, emit, trace_path=None):
    originals = []
    counts = {}
    traced = False

    def sync():
        if device != 'cpu':
            torch.cuda.synchronize(int(device[5:]))

    def memory():
        if device == 'cpu':
            return {}
        return dict(allocated_bytes=torch.cuda.memory_allocated(int(device[5:])),
                    reserved_bytes=torch.cuda.memory_reserved(int(device[5:])),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(int(device[5:])),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(int(device[5:])))

    def wrap(owner, method, stage, synchronize=True):
        original = getattr(owner, method)
        had_local = method in vars(owner)
        originals.append((owner, method, original, had_local))

        @wraps(original)
        def measured(*args, **kwargs):
            nonlocal traced
            counts[stage] = counts.get(stage, 0) + 1
            index = counts[stage]
            if synchronize:
                sync()
                if device != 'cpu':
                    torch.cuda.reset_peak_memory_stats(int(device[5:]))
            start = time.perf_counter()
            profile = None
            if stage == 'transformer' and trace_path is not None and not traced:
                traced = True
                activities = [torch.profiler.ProfilerActivity.CPU]
                if device != 'cpu':
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                profile = torch.profiler.profile(activities=activities, record_shapes=False,
                    profile_memory=False, with_stack=False)
            emit('image.phase.begin', stage=stage, index=index, **memory())
            status = 'failed'
            try:
                with profile if profile is not None else nullcontext():
                    result = original(*args, **kwargs)
                    if synchronize:
                        sync()
                status = 'completed'
                return result
            finally:
                elapsed = time.perf_counter() - start
                emit('image.phase.end', stage=stage, index=index, seconds=elapsed, status=status, **memory())
                if profile is not None:
                    profile.export_chrome_trace(str(trace_path))
                    # CPU copy dispatch and actual GPU memcpy are distinct. The
                    # Chrome trace retains timestamps/overlap and kernel devices.
                    copies = [dict(name=row.key, count=row.count,
                        cpu_us=row.self_cpu_time_total,
                        device_us=getattr(row, 'self_device_time_total', 0))
                        for row in profile.key_averages()
                        if 'copy' in row.key.lower() or 'memcpy' in row.key.lower() or 'to_copy' in row.key]
                    emit('image.transfer.profile', trace=str(trace_path), events=copies)
        setattr(owner, method, measured)

    try:
        sync()
        if device != 'cpu':
            torch.cuda.reset_peak_memory_stats(int(device[5:]))
        wrap(pipeline, 'encode_prompt', 'prompt_encoding')
        wrap(pipeline.transformer, 'forward', 'transformer')
        wrap(pipeline.vae, 'decode', 'vae_decode')
        # Count tiled decoder forwards without forcing a sync for each tile.
        wrap(pipeline.vae.decoder, 'forward', 'vae_tile', synchronize=False)
        wrap(pipeline.image_processor, 'postprocess', 'postprocess')
        yield
    finally:
        for owner, method, original, had_local in reversed(originals):
            if had_local:
                setattr(owner, method, original)
            else:
                delattr(owner, method)
        emit('image.profile.complete', counts=dict(counts), **memory())
