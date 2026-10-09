"""Short diagnostic body for a reviewed external supervisor; default is read-only.

No imports of Torch/CUDA at module scope, no model checkpoint or production queue.
The external supervisor owns admission, sensors, death fencing and hard deadlines.
"""
import ctypes
import importlib
import os
from pathlib import Path
import signal
import sys
from uuid import UUID

import hashlib
import json
import time

PLAN = dict(seconds=5, maximum_iterations=4096, shape=[2048, 2048], dtype='bfloat16',
            allocator_bytes=512*1024**2, gpu_execution=False, external_supervisor_required=True)


def compare(control, candidate):
    for key in ('device', 'gpu_uuid', 'output_sha256', 'shape', 'dtype'):
        if control[key] != candidate[key]:
            raise ValueError('Control/candidate numerical identity differs: '+key)
    if control['policy'] != 'control' or candidate['policy'] != 'blocking':
        raise ValueError('Expected fresh control and blocking cases')
    if control['flags']['before'] & 7 == 4:
        return dict(numerical_equal=True, intervention=False, verdict='inconclusive: control already blocking')
    if candidate['flags']['after'] & 7 != 4:
        raise ValueError('Candidate blocking flags unconfirmed')
    return dict(numerical_equal=True, intervention=True,
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
    actual_uuid='GPU-'+str(UUID(str(torch.cuda.get_device_properties(device).uuid).removeprefix('GPU-')))
    expected=json.loads(os.environ['KADAN_EXPECTED_GPU_UUIDS'])
    if len(expected)!=2 or len(set(expected))!=2 or actual_uuid!=expected[device]:
        raise RuntimeError('Physical GPU UUID binding differs')
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
        return dict(device=device, gpu_uuid=actual_uuid, policy=policy, flags=flags, shape=PLAN['shape'], dtype=PLAN['dtype'],
                    iterations=iterations, wall_seconds=wall, process_cpu_seconds=process,
                    main_cpu_seconds=main, output_sha256=hashlib.sha256(raw).hexdigest())


def owned_child(policy, device, parent, commit):
    # Fence before importing Torch or any code that might initialize CUDA.
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Parent-death fence failed')
    if os.getppid() != parent or sorted(os.sched_getaffinity(0)) != [0,8]:
        raise RuntimeError('Owned probe parent/affinity differs')
    torch=importlib.import_module('torch')
    result=measure(torch,device,policy)
    result.update(commit=commit,pid=os.getpid())
    output=Path('/evidence')/f'rank-{device}.json'
    with output.open('x') as stream: json.dump(result,stream)


if __name__ == '__main__':
    if len(sys.argv)==6 and sys.argv[1]=='--owned-child':
        owned_child(sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),sys.argv[5])
    elif len(sys.argv)==1:
        print(json.dumps(PLAN, sort_keys=True))
    else:
        raise ValueError('Use the reviewed launch_cuda_wait.py supervisor')
