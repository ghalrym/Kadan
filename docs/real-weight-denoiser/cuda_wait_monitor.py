"""Independent fail-closed thermal sampling and read-only Linux task counters."""
import json
import os
from pathlib import Path
import signal
import threading
import time

from thermal_guard import check_cpu
from nvml_sensor import NVMLReader


class ThermalWatch:
    interval = .5
    freshness = 1.0
    guard_interval = .05

    def __init__(self, evidence, gpus):
        self.evidence, self.gpus = Path(evidence), tuple(gpus)
        self.stop_event, self.ready = threading.Event(), threading.Event()
        self.error = None
        self.lock = threading.Lock()
        self.deadline = None
        self.last_valid = None
        self.reader = NVMLReader(self.gpus)
        self.thread = threading.Thread(target=self._run, name='cuda-wait-thermal', daemon=True)
        self.guard = threading.Thread(target=self._guard, name='cuda-wait-freshness', daemon=True)

    def start(self):
        self.deadline = time.monotonic() + self.freshness
        self.guard.start()
        self.thread.start()
        if not self.ready.wait(2):
            self._fail(RuntimeError('Thermal monitor did not become ready'))
        self.check()

    def check(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self._fail(RuntimeError('Thermal sample stale'))
        if self.error is not None: raise RuntimeError('Independent thermal monitor: '+self.error)
        if not self.thread.is_alive(): raise RuntimeError('Thermal monitor stopped unexpectedly')

    def close(self):
        self.stop_event.set()
        end = time.monotonic() + 1
        for thread in (self.thread, self.guard):
            if thread.ident is not None:
                thread.join(timeout=max(0, end-time.monotonic()))
        self.reader.close(end)
        if self.thread.is_alive() or self.guard.is_alive():
            raise RuntimeError('Thermal monitor teardown unconfirmed')
        if self.error is not None: raise RuntimeError('Independent thermal monitor: '+self.error)

    def _fail(self, exc):
        # Abort before evidence I/O: a blocked/full filesystem cannot delay cleanup.
        with self.lock:
            if self.error is not None: return
            self.error = f'{type(exc).__name__}: {exc}'[:1000]
            self.stop_event.set()
            self.ready.set()
        os.kill(os.getpid(), signal.SIGUSR1)

    def _guard(self):
        # No subprocesses, sensor reads or evidence I/O on this thread.
        while not self.stop_event.wait(self.guard_interval):
            if self.deadline is not None and time.monotonic() >= self.deadline:
                self._fail(RuntimeError('Thermal sample stale: acquisition or evidence blocked'))
                return

    def _run(self):
        next_sample = time.monotonic()
        if self.deadline is None: self.deadline = next_sample + self.freshness
        started = query_started = None
        row = None
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                query_started = None
                row = None
                deadline = self.deadline
                if started >= deadline: raise RuntimeError('Thermal sample stale')
                peak = check_cpu(self.evidence, 'independent-wait-watchdog')
                query_started = time.monotonic()
                remaining = deadline-query_started
                if remaining <= 0: raise RuntimeError('Thermal sample stale before GPU query')
                result = self.reader.sample(timeout=remaining)
                acquired = time.monotonic()
                if acquired >= deadline: raise RuntimeError('Thermal sample stale after GPU query')
                rows=result['gpu']
                if set(rows)!=set(self.gpus):raise RuntimeError('NVML GPU identity mismatch')
                accepted=bool(self.gpus) and all(g in rows and 0 <= rows[g]['temperature'] < 90
                    and rows[g]['free']>=256 and rows[g]['used']>=0 for g in self.gpus)
                memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
                if int(memory['MemAvailable'].split()[0])<16*1024**2:raise RuntimeError('Host free memory guard')
                row=dict(monotonic=started, acquisition_started=started, acquisition_finished=acquired,
                    acquisition_seconds=acquired-started, query_seconds=acquired-query_started,
                    sample_age_seconds=acquired-started, previous_sample_age_seconds=None if self.last_valid is None
                    else acquired-self.last_valid, freshness_deadline=deadline,
                    cpu_temperature=peak,gpu=rows,accepted=accepted, sensor_backend='persistent-nvml',
                    sensor_timing={k:v for k,v in result.items() if k!='gpu'})
                if not accepted: raise RuntimeError('GPU sensor/headroom guard')
                with (self.evidence/'watchdog.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                # Persisted, validated and still fresh. Acquisition start is conservative.
                with self.lock:
                    if self.error is not None or self.stop_event.is_set(): return
                    if time.monotonic() >= min(deadline, started+self.freshness):
                        raise RuntimeError('Thermal sample stale after evidence write')
                    self.last_valid = started
                    self.deadline = started+self.freshness
                self.ready.set()
                next_sample += self.interval
                self.stop_event.wait(max(0,next_sample-time.monotonic()))
        except BaseException as exc:
            self._fail(exc)
            try:
                (self.evidence/'watchdog-error.json').write_text(json.dumps(dict(error=self.error,
                    failed_at=time.monotonic(), acquisition_started=started, query_started=query_started,
                    acquisition_seconds=None if started is None else time.monotonic()-started,
                    sample_age_seconds=None if self.last_valid is None else time.monotonic()-self.last_valid,
                    rejected_sample=row, last_valid=self.last_valid, deadline=self.deadline)))
            except OSError:
                pass
        finally:
            try:self.reader.close()
            except Exception as exc:self._fail(exc)


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
