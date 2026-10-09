"""Independent fail-closed thermal sampling and read-only Linux task counters."""
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from thermal_guard import check_cpu


class ThermalWatch:
    def __init__(self, evidence, gpus):
        self.evidence, self.gpus = Path(evidence), gpus
        self.stop_event, self.ready = threading.Event(), threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name='cuda-wait-thermal', daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready.wait(2): raise RuntimeError('Thermal monitor did not become ready')
        self.check()

    def check(self):
        if self.error is not None: raise RuntimeError('Independent thermal monitor: '+self.error)
        if not self.thread.is_alive(): raise RuntimeError('Thermal monitor stopped unexpectedly')

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=1)
        if self.thread.is_alive(): raise RuntimeError('Thermal monitor teardown unconfirmed')

    def _run(self):
        previous = None
        next_sample = time.monotonic()
        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                if previous is not None and now-previous > 1:
                    raise RuntimeError('Thermal cadence exceeded one second')
                previous = now
                peak = check_cpu(self.evidence, 'independent-wait-watchdog')
                # This worker never waits on Docker or telemetry collection.
                result = subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free,temperature.gpu',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=.35)
                rows={}
                for line in result.stdout.splitlines():
                    uuid,used,free,temp=[x.strip() for x in line.split(',')]
                    rows[uuid]=dict(used=int(used),free=int(free),temperature=int(temp))
                accepted=all(g in rows and rows[g]['temperature']<90 and rows[g]['free']>=256 for g in self.gpus)
                with (self.evidence/'watchdog.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(monotonic=now,cpu_temperature=peak,gpu=rows,accepted=accepted))+'\n')
                if not accepted:
                    raise RuntimeError('GPU sensor/headroom guard')
                memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
                if int(memory['MemAvailable'].split()[0])<16*1024**2:raise RuntimeError('Host free memory guard')
                self.ready.set()
                next_sample += .5
                self.stop_event.wait(max(0,next_sample-time.monotonic()))
        except BaseException as exc:
            self.error=f'{type(exc).__name__}: {exc}'[:1000]
            try:
                (self.evidence/'watchdog-error.json').write_text(json.dumps(dict(error=self.error)))
            finally:
                self.ready.set()
                # Evidence I/O failure must not suppress the abort signal.
                os.kill(os.getpid(),signal.SIGUSR1)


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
