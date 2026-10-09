"""Bounded two-process control transport; no request queue and no tensor IPC."""
import importlib
import json
import os
from pathlib import Path
import selectors
import signal
import socket
import subprocess
import sys
import tempfile
import time

from api.inference.resources import ResourceCancelled

MAX_FRAME = 65536


def encode(value):
    data = json.dumps(value, allow_nan=False, separators=(',', ':')).encode() + b'\n'
    if len(data) > MAX_FRAME:
        raise ValueError('Rank control frame exceeds 64 KiB')
    return data


def receive_blocking(connection):
    data = bytearray()
    while not data.endswith(b'\n'):
        part = connection.recv(1)
        if not part:
            raise EOFError('Rank controller closed')
        data.extend(part)
        if len(data) > MAX_FRAME:
            raise ValueError('Oversized rank command')
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError('Rank command must be an object')
    return value


class ProcessRanks:
    def __init__(self, path, budget, *, worker_module='api.inference.image.rank_worker', guard=None, memory_probe=None):
        self.path, self.budget, self.worker_module = Path(path), budget, worker_module
        self.guard = guard or self._guard
        self.cooldown = guard is None
        self.processes, self.connections = [], []
        self.directory = None
        self.last_guard = 0.
        self.memory_probe = memory_probe or self._device_usage
        self.process_probe = self._compute_processes if memory_probe is None else None
        self.baseline_processes = set()
        self.owned_processes = None
        self.baseline = None
        self.uuids = None
        self.gone = set()

    def start(self, session, devices, deadline, cancel):
        if self.processes or self.directory is not None:
            raise RuntimeError('Previous rank ownership has not been reaped')
        if self.cooldown:
            cool = 0
            admission_end = min(deadline, time.monotonic()+30)
            while cool < 5:
                if cancel is not None and cancel.is_set():
                    raise ResourceCancelled('Rank cooldown cancelled')
                if time.monotonic() >= admission_end:
                    raise TimeoutError('CPU did not reach five cool admission samples')
                cool = cool+1 if self._cpu_peak() < 60 else 0
                if cool < 5:
                    time.sleep(2)
        self.baseline = self.memory_probe()
        if set(self.baseline) != set(devices):
            raise RuntimeError("Physical device ownership is unavailable")
        self.baseline_processes = set(self.process_probe()) if self.process_probe else set()
        self.owned_processes = None
        self.gone.clear()
        self.directory = tempfile.TemporaryDirectory(prefix='kadan-image-ranks-')
        cpus = sorted(os.sched_getaffinity(0))[:2]
        for rank, device in enumerate(devices):
            if cancel is not None and cancel.is_set():
                raise ResourceCancelled('Rank startup cancelled')
            if time.monotonic() >= deadline:
                raise TimeoutError('Rank startup deadline exhausted')
            parent, child = socket.socketpair()
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                NUMEXPR_NUM_THREADS='1', GLOO_SOCKET_IFNAME='lo', PYTHONDONTWRITEBYTECODE='1')
            config = dict(session=session, rank=rank, device=device, devices=list(devices),
                checkpoint=str(self.path), directory=self.directory.name, cpus=cpus,
                execution_bytes=self.budget.execution_bytes, parent_pid=os.getpid())
            try:
                process = subprocess.Popen([sys.executable, '-m', self.worker_module,
                    str(child.fileno()), json.dumps(config)], pass_fds=(child.fileno(),),
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True, env=env)
            except BaseException:
                parent.close()
                raise
            finally:
                child.close()
            self.processes.append(process)
            self.connections.append(parent)
            parent.setblocking(False)

    def _device_usage(self):
        # Optional Torch access is only used after shared admission; resource
        # probing already establishes the API's CUDA visibility mapping.
        if self.uuids is None:
            torch = importlib.import_module('torch')
            self.uuids = {d:str(torch.cuda.get_device_properties(d).uuid) for d in self.budget.devices}
        result = subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True,text=True,timeout=2,check=True)
        rows = {parts[0].strip():int(parts[1].strip())*1024**2
            for parts in (line.split(',') for line in result.stdout.splitlines())}
        return {d:rows[uuid] for d,uuid in self.uuids.items()}

    def _compute_processes(self):
        result = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2,check=True)
        rows = {}
        for line in result.stdout.splitlines():
            uuid,pid,memory = [part.strip() for part in line.split(',')]
            if uuid in self.uuids.values():
                rows[(uuid,int(pid))] = int(memory)*1024**2
        return rows

    def _confirm_owners(self):
        if self.process_probe is None:
            return
        rows = self.process_probe()
        owners = set(rows)-self.baseline_processes
        if len(owners)!=2 or {uuid for uuid,pid in owners}!=set(self.uuids.values()):
            raise RuntimeError('Cannot uniquely identify both physical GPU rank owners')
        self.owned_processes = owners

    def _owned_fit(self, allowance, *, stopped=False):
        if self.process_probe is None:
            return self._physical(allowance)
        rows = self.process_probe()
        owners = self.owned_processes
        if stopped:
            return not ((owners if owners is not None else set(rows)-self.baseline_processes) & set(rows))
        if owners is None:
            return self._physical(allowance)
        return owners<=set(rows) and all(rows[owner]<=allowance for owner in owners)

    def _physical(self, allowance):
        if self.baseline is None:
            return True
        current = self.memory_probe()
        return set(current)==set(self.baseline) and all(
            current[d] <= baseline+allowance for d,baseline in self.baseline.items())

    @staticmethod
    def _cpu_peak():
        temperatures = []
        for path in Path('/sys/class/hwmon').glob('hwmon*/temp*_input'):
            if (path.parent/'name').read_text().strip() in ('k10temp', 'coretemp'):
                temperatures.append(int(path.read_text()) / 1000)
        if not temperatures:
            raise RuntimeError('CPU thermal sensors unavailable')
        return max(temperatures)

    def _guard(self):
        # RSS double-counts shared mappings conservatively. No swap is invented
        # as extra capacity. Unknown readings fail closed while work is active.
        if not self._owned_fit(self.budget.execution_bytes+self.budget.context_bytes):
            raise RuntimeError("Physical rank GPU envelope exceeded")
        rss = 0
        for process in self.processes:
            status = Path(f'/proc/{process.pid}/status').read_text()
            rss += int(next(l.split()[1] for l in status.splitlines() if l.startswith('VmRSS:'))) * 1024
        if rss > self.budget.host_bytes:
            raise RuntimeError('Rank host memory exceeded admission')
        if self._cpu_peak() >= 80:
            raise RuntimeError('Image CPU thermal guard reached')
        output = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2, check=True)
        values = [float(value) for value in output.stdout.splitlines()]
        if not values or max(values) >= 90:
            raise RuntimeError('Image GPU thermal guard reached or unavailable')

    def exchange(self, command, deadline, cancel):
        payload = encode(command)
        pending = {i: bytearray(payload) for i in range(2)}
        buffers = {i: bytearray() for i in range(2)}
        replies = {}
        with selectors.DefaultSelector() as selector:
            for rank, connection in enumerate(self.connections):
                selector.register(connection, selectors.EVENT_READ | selectors.EVENT_WRITE, rank)
            while len(replies) != 2:
                if cancel is not None and cancel.is_set():
                    raise ResourceCancelled('Rank transaction cancelled')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Rank transaction timed out')
                if len(self.processes) != 2 or any(p.poll() is not None for p in self.processes):
                    raise RuntimeError('Image rank exited; both ranks must be reaped')
                if time.monotonic() - self.last_guard >= 1:
                    self.guard()
                    self.last_guard = time.monotonic()
                for key, mask in selector.select(min(.05, max(0, deadline-time.monotonic()))):
                    rank = key.data
                    if mask & selectors.EVENT_WRITE and pending[rank]:
                        sent = key.fileobj.send(pending[rank])
                        del pending[rank][:sent]
                        if not pending[rank]:
                            selector.modify(key.fileobj, selectors.EVENT_READ, rank)
                    if mask & selectors.EVENT_READ:
                        data = key.fileobj.recv(MAX_FRAME + 1)
                        if not data:
                            raise EOFError('Rank closed without both-rank completion')
                        buffers[rank].extend(data)
                        if len(buffers[rank]) > MAX_FRAME:
                            raise ValueError('Oversized rank acknowledgement')
                        if b'\n' in buffers[rank]:
                            frame, extra = buffers[rank].split(b'\n', 1)
                            if extra or pending[rank]:
                                raise ValueError('Unsolicited rank acknowledgement')
                            value = json.loads(frame)
                            if not isinstance(value, dict):
                                raise ValueError('Invalid rank acknowledgement')
                            replies[rank] = value
                            selector.unregister(key.fileobj)
        if command["operation"] == "ready":
            self._confirm_owners()
        if command["operation"] == "park" and not self._owned_fit(self.budget.context_bytes):
            raise RuntimeError("Physical GPU residency did not return to the parked envelope")
        if command["operation"] == "park" and self.directory is not None:
            for path in Path(self.directory.name).glob("rendezvous-*"):
                path.unlink(missing_ok=True)
        return [replies[0], replies[1]]

    def stop(self, deadline):
        # Kill whole owned process groups; never infer physical cleanup from EOF.
        for process in self.processes:
            if process.pid in self.gone:
                continue
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for process in self.processes:
            try:
                process.wait(timeout=max(.001, deadline-time.monotonic()))
                os.killpg(process.pid, 0)
                return False  # An owned descendant is still present.
            except ProcessLookupError:
                self.gone.add(process.pid)
            except subprocess.TimeoutExpired:
                return False
        try:
            while self.processes and not self._owned_fit(64*1024**2, stopped=True):
                if time.monotonic() >= deadline:
                    return False
                time.sleep(.05)
        except Exception:
            return False
        for connection in self.connections:
            connection.close()
        self.processes.clear()
        self.connections.clear()
        self.baseline = None
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None
        return True

    def output(self):
        if self.directory is None:
            raise RuntimeError('Rank output ownership is closed')
        return Path(self.directory.name) / 'output.png'
