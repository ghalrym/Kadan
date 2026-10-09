"""Short two-rank probe; reuse baseline lock-through-reap/container ownership."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def cleanup(children, deadline):
    # Signal every owned group before waiting; no per-child deadline reset.
    for child in children:
        try: os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    for child in children:
        child.wait(timeout=max(.001, deadline-time.monotonic()))


def main():
    policy, commit = sys.argv[1:]
    if policy not in ('control', 'blocking'):
        raise ValueError('Invalid wait policy')
    cpus = sorted(os.sched_getaffinity(0))
    if cpus != [0, 8]: raise ValueError('Reviewed CPU placement differs')
    if Path('/sys/fs/cgroup/memory.max').read_text().strip() != str(8*1024**3):
        raise ValueError('Expected 8 GiB memory cap')
    if Path('/sys/fs/cgroup/memory.swap.max').read_text().strip() != '0':
        raise ValueError('Swap must be disabled')
    shm=os.statvfs('/dev/shm')
    if shm.f_blocks*shm.f_frsize != 1024**3: raise ValueError('Expected 1 GiB SHM')
    def interrupted(signum, frame): raise InterruptedError(f'Wait supervisor signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    children=[]; status=1; started=time.monotonic()
    with open('/models/inference.lock','rb') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for rank in (0,1):
                children.append(subprocess.Popen([sys.executable,'/probe/cuda_wait_probe.py',
                    '--owned-child',policy,str(rank),str(os.getpid()),commit],start_new_session=True))
            deadline=started+30
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None,0) for child in children):
                    raise RuntimeError('A wait probe rank failed')
                if time.monotonic() >= deadline: raise TimeoutError('Wait probe stage exceeded 30 seconds')
                time.sleep(.05)
            if any(child.returncode != 0 for child in children): raise RuntimeError('Wait probe failed')
            status=0
        finally:
            cleanup(children,time.monotonic()+30)
            Path('/evidence/supervisor.json').write_text(json.dumps(dict(returncode=status,
                policy=policy,commit=commit,elapsed_seconds=time.monotonic()-started,
                queue_inode=os.fstat(lease.fileno()).st_ino,children=[p.pid for p in children],children_reaped=True)))
    return status


if __name__ == '__main__': sys.exit(main())
