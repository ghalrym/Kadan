"""Baseline PID1 keeps the existing queue inode locked through child teardown."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    case, commit = sys.argv[1:]
    if case not in ('A', 'B'):
        raise ValueError('Unknown baseline case')
    quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    if quota == 'max' or int(quota) > 2*int(period):
        raise ValueError('Baseline requires at most two CPUs')
    if int(Path('/sys/fs/cgroup/memory.max').read_text()) > 128*1024**3:
        raise ValueError('Baseline requires at most 128 GiB RAM')
    if Path('/sys/fs/cgroup/memory.swap.max').read_text().strip() != '0':
        raise ValueError('Baseline swap must be disabled')
    shm = os.statvfs('/dev/shm')
    if shm.f_blocks*shm.f_frsize != 1024**3:
        raise ValueError('Baseline requires explicit 1 GiB SHM')
    stage = int(os.environ['KADAN_STAGE_SECONDS'])
    if not 1 <= stage <= 900:
        raise ValueError('Invalid stage deadline')
    with open('/models/inference.lock', 'rb') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = Path('/evidence/baseline')
        output.mkdir()
        child = subprocess.Popen([sys.executable, '/probe/api_baseline.py', '--case', case,
            '--commit', commit, '--output', str(output)], start_new_session=True)
        def interrupted(signum, frame):
            raise InterruptedError(f'Baseline supervisor signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        started = time.monotonic()
        status = 1
        try:
            status = child.wait(timeout=stage)
        finally:
            # The container remains the outer fence for any surviving descendants.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=10)
            Path('/evidence/supervisor.json').write_text(json.dumps(dict(
                returncode=status, elapsed_seconds=time.monotonic()-started, child_reaped=True,
                queue_inode=os.fstat(lease.fileno()).st_ino, shm_bytes=shm.f_blocks*shm.f_frsize)))
        return status


if __name__ == '__main__':
    sys.exit(main())
