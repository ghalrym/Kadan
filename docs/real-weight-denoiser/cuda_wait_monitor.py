"""Read-only Linux task counters and cleanup helpers."""
import json
import os
from pathlib import Path
import time


def cgroup_identity(pid, proc=Path('/proc'), sysfs=Path('/sys/fs/cgroup')):
    entries=(proc/str(pid)/'cgroup').read_text().splitlines()
    relative=next(line[3:] for line in entries if line.startswith('0::'))
    path=sysfs/relative.lstrip('/')
    if not path.resolve().is_relative_to(sysfs.resolve()):raise ValueError('Invalid cgroup path')
    values={name:(path/name).read_text().strip() for name in ('cpu.max','cpuset.cpus','cpuset.cpus.effective','memory.max','memory.swap.max')}
    return path,dict(pid1=pid,path=str(path),**values)


def task_counters(cgroup, proc=Path('/proc')):
    rows=[]
    for pid in (cgroup/'cgroup.procs').read_text().split():
        try:
            raw=(proc/pid/'stat').read_text();fields=raw.rsplit(')',1)[1].split();start=int(fields[19])
            threads=[]
            for task in (proc/pid/'task').iterdir():
                try:
                    stat=(task/'stat').read_text();parts=stat.rsplit(')',1)[1].split()
                    status=(task/'status').read_text()
                    allowed=next(line.split(':',1)[1].strip() for line in status.splitlines() if line.startswith('Cpus_allowed_list:'))
                    threads.append(dict(tid=int(task.name),name=stat[stat.index('(')+1:stat.rindex(')')],
                        start_ticks=int(parts[19]),utime=int(parts[11]),stime=int(parts[12]),cpu=int(parts[36]),allowed=allowed))
                except (FileNotFoundError,ProcessLookupError):continue
            rows.append(dict(pid=int(pid),start_ticks=start,threads=threads))
        except (FileNotFoundError,ProcessLookupError):continue
    return dict(monotonic=time.monotonic(),clock_ticks=os.sysconf('SC_CLK_TCK'),processes=rows)


def close_watchdog(monitor):
    """Return a late failure without I/O or exceptions blocking owner teardown."""
    if monitor is None:return None
    try:monitor.close()
    except Exception as exc:return str(exc)
    return None


def publish_watchdog_error(evidence, error):
    """Only called after physical cleanup and safe restoration; best effort."""
    try:(Path(evidence)/'watchdog-cleanup-error.json').write_text(json.dumps(dict(error=error)))
    except OSError:pass
