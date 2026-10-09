"""Explicit CPU pool limits before model/metric operations; numerical engine unchanged."""
import json
import os
from pathlib import Path

import torch
import replay_holdout


def configure_threads():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    state=dict(intraop=torch.get_num_threads(),interop=torch.get_num_interop_threads(),
        affinity_cpus=len(os.sched_getaffinity(0)),cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        replay_commit=os.environ['KADAN_REVIEWED_COMMIT'],capture_commit=os.environ['KADAN_CAPTURE_COMMIT'])
    quota,period=state['cpu_max'].split()
    if quota=='max' or int(quota)>2*int(period):raise RuntimeError('Replay cgroup CPU quota exceeds two CPUs')
    if state['intraop']!=1 or state['interop']!=1:raise RuntimeError('CPU pool bound did not apply')
    (Path('/evidence')/f'cpu-threads-rank-{os.environ["LOCAL_RANK"]}.json').write_text(json.dumps(state,indent=2))
    return state


if __name__=='__main__':
    configure_threads()
    replay_holdout.main()
