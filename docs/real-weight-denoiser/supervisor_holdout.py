"""Container PID1: hold the existing queue lease until every rank has stopped."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    # Open the existing inode read-only; never replace/unlink the queue lease.
    with open('/models/inference.lock', 'rb') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = 1
        stage_seconds = int(os.environ['KADAN_STAGE_SECONDS'])
        assert 1 <= stage_seconds <= 900
        assert sys.argv[1:] in (['holdout-capture'],['holdout-replay'])
        step=os.environ['KADAN_HELDOUT_STEP'];assert step in ('20','39')
        if sys.argv[1]=='holdout-capture':
            command=[sys.executable,'/probe/capture_holdout.py','--step',step]
        else:
            command=[sys.executable,'-m','torch.distributed.run','--nnodes=1',
                '--node_rank=0','--master_addr=127.0.0.1','--master_port=29500',
                '--nproc_per_node=2','--max_restarts=0','/probe/replay_holdout.py']
        child = subprocess.Popen(command, start_new_session=True)
        def stop(signum, frame):
            raise InterruptedError(f'Supervisor received signal {signum}')
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        started = time.monotonic()
        try:
            status = child.wait(timeout=stage_seconds)
        finally:
            # torchrun propagates rank errors; independently reap its entire group.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
            # A surviving orphan may not be a direct child. Kill the process group
            # even when torchrun has already exited. Container exit is a final fence.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            Path('/evidence/supervisor.json').write_text(json.dumps(dict(
                returncode=status, elapsed_s=time.monotonic()-started, torchrun_reaped=child.poll() is not None)))
        return status


if __name__ == '__main__':
    sys.exit(main())
