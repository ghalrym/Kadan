"""Short diagnostic body for a reviewed external supervisor; default is read-only.

No imports of Torch/CUDA at module scope, no model checkpoint or production queue.
The external supervisor owns admission, sensors, death fencing and hard deadlines.
"""
import hashlib
import json
import time

PLAN = dict(seconds=5, maximum_iterations=4096, shape=[2048, 2048], dtype='bfloat16',
            allocator_bytes=512*1024**2, gpu_execution=False, external_supervisor_required=True)


def compare(control, candidate):
    for key in ('device', 'output_sha256', 'shape', 'dtype'):
        if control[key] != candidate[key]:
            raise ValueError('Control/candidate numerical identity differs: '+key)
    if control['policy'] != 'control' or candidate['policy'] != 'blocking':
        raise ValueError('Expected fresh control and blocking cases')
    return dict(numerical_equal=True,
                control_cpu_per_wall=control['process_cpu_seconds']/control['wall_seconds'],
                candidate_cpu_per_wall=candidate['process_cpu_seconds']/candidate['wall_seconds'])


def measure(torch, device, policy):
    """Called only in an owned child after supervisor admission, before model work."""
    if policy not in ('control', 'blocking'):
        raise ValueError('Unknown wait policy')
    # Optional worker-only dependency; the read-only plan imports no API runtime.
    from api.inference.image.cuda_wait import Runtime, configure_blocking_sync
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(device)
    if policy == 'blocking':
        flags = configure_blocking_sync(device)
    else:
        runtime = Runtime()
        if runtime.device() != device:
            raise RuntimeError('Control device differs')
        flags = dict(device=device, runtime_version=runtime.version(),
                     before=runtime.flags(), after=runtime.flags())
    torch.cuda.set_per_process_memory_fraction(
        PLAN['allocator_bytes']/torch.cuda.get_device_properties(device).total_memory, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        # Small exactly representable values avoid overflow; fixed CPU construction.
        row = ((torch.arange(2048, dtype=torch.float32) % 17)-8)/16
        operand = row.repeat(2048, 1).to(dtype=torch.bfloat16, device=device)
        other = operand.transpose(0, 1).contiguous()
        result = operand @ other
        torch.cuda.synchronize(device)
        if not bool(torch.isfinite(result).all()):
            raise ValueError('Nonfinite diagnostic warmup')
        started, cpu, main_cpu = time.monotonic(), time.process_time(), time.thread_time()
        iterations = 0
        while iterations < PLAN['maximum_iterations']:
            result = operand @ other
            if not bool(torch.isfinite(result).all()):
                raise ValueError('Nonfinite diagnostic result')
            iterations += 1
            if time.monotonic()-started >= PLAN['seconds']:
                break
        torch.cuda.synchronize(device)
        wall, process, main = time.monotonic()-started, time.process_time()-cpu, time.thread_time()-main_cpu
        raw = result.view(torch.uint8).cpu().numpy().tobytes()
        return dict(device=device, policy=policy, flags=flags, shape=PLAN['shape'], dtype=PLAN['dtype'],
                    iterations=iterations, wall_seconds=wall, process_cpu_seconds=process,
                    main_cpu_seconds=main, output_sha256=hashlib.sha256(raw).hexdigest())


if __name__ == '__main__':
    print(json.dumps(PLAN, sort_keys=True))
