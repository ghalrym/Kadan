"""Persistent read-only NVML helper. No CUDA imports, contexts or model execution."""
import ctypes
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
import threading
from uuid import UUID

MAX_REPLY = 8192


class Memory(ctypes.Structure):
    _fields_ = [('version',ctypes.c_uint),('total',ctypes.c_ulonglong),
                ('reserved',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]


class NVML:
    def __init__(self, uuids, library=None):
        self.library = library or ctypes.CDLL('libnvidia-ml.so.1')
        self.handles = {}
        def bind(name, args):
            fn=getattr(self.library,name);fn.argtypes=args;fn.restype=ctypes.c_int;return fn
        self.init=bind('nvmlInit_v2',[])
        self.handle=bind('nvmlDeviceGetHandleByUUID',[ctypes.c_char_p,ctypes.POINTER(ctypes.c_void_p)])
        self.identity=bind('nvmlDeviceGetUUID',[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_uint])
        self.memory=bind('nvmlDeviceGetMemoryInfo_v2',[ctypes.c_void_p,ctypes.POINTER(Memory)])
        self.temperature=bind('nvmlDeviceGetTemperature',[ctypes.c_void_p,ctypes.c_uint,ctypes.POINTER(ctypes.c_uint)])
        self.shutdown=bind('nvmlShutdown',[])
        self.call(self.init)
        for uuid in uuids:
            if uuid!='GPU-'+str(UUID(uuid.removeprefix('GPU-'))) or uuid in self.handles:
                raise ValueError('Invalid/duplicate physical GPU UUID')
            handle=ctypes.c_void_p();self.call(self.handle,uuid.encode(),ctypes.byref(handle))
            actual=ctypes.create_string_buffer(96);self.call(self.identity,handle,actual,len(actual))
            if actual.value.decode()!=uuid:raise ValueError('NVML GPU identity mismatch')
            self.handles[uuid]=handle
        if not self.handles:raise ValueError('No physical GPUs')

    @staticmethod
    def call(fn,*args):
        started=time.monotonic();code=fn(*args)
        if code:raise RuntimeError(f'{fn.__name__} failed with NVML status {code}')
        return time.monotonic()-started

    def sample(self):
        rows={};timings={}
        for uuid,handle in self.handles.items():
            memory=Memory();memory.version=ctypes.sizeof(Memory)|(2<<24)
            temperature=ctypes.c_uint()
            mt=self.call(self.memory,handle,ctypes.byref(memory))
            tt=self.call(self.temperature,handle,0,ctypes.byref(temperature))
            if not (memory.reserved<=memory.total and memory.used+memory.free<=memory.total):
                raise ValueError('Invalid NVML memory accounting')
            # Preserve v2 raw accounting; do not subtract reserved a second time.
            rows[uuid]=dict(used=memory.used//1048576,free=memory.free//1048576,
                           reserved=memory.reserved//1048576,temperature=temperature.value)
            timings[uuid]=dict(memory_seconds=mt,temperature_seconds=tt)
        return dict(gpu=rows,nvml_call_seconds=timings)


class NVMLReader:
    """One request at a time; a stuck native driver call is killed in close()."""
    def __init__(self, uuids, command=None):
        self.command=command or [sys.executable,str(Path(__file__).resolve()),json.dumps(list(uuids)),str(os.getpid())]
        self.process=None;self.closed=False;self.sequence=0
        self.state_lock=threading.Lock()

    def sample(self, *, timeout):
        with self.state_lock:
            if self.closed:raise RuntimeError('NVML reader closed')
            process=self.process
        started=time.monotonic();deadline=started+timeout;spawn_seconds=0
        if process is None:
            # Popen can block. Never hold the state lock across process/pipe I/O.
            process=subprocess.Popen(self.command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,bufsize=0,start_new_session=True)
            spawn_seconds=time.monotonic()-started
            with self.state_lock:
                closed=self.closed
                if not closed:self.process=process
            if closed:
                # close() may have returned while Popen was still in progress.
                # The spawning thread owns this unpublished child and must reap
                # it before any command write or propagation to error logging.
                self._reap(process,time.monotonic()+.2)
                raise RuntimeError('NVML reader closed during startup')
            os.set_blocking(process.stdout.fileno(),False)
        self.sequence+=1
        # At most one tiny command is outstanding, so the pipe cannot fill.
        process.stdin.write((str(self.sequence)+'\n').encode())
        data=bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout,selectors.EVENT_READ)
            while b'\n' not in data:
                remaining=deadline-time.monotonic()
                if remaining<=0 or not selector.select(remaining):raise TimeoutError('NVML acquisition deadline')
                chunk=os.read(process.stdout.fileno(),MAX_REPLY+1-len(data))
                if not chunk:raise RuntimeError('NVML helper exited without sample')
                data.extend(chunk)
                if len(data)>MAX_REPLY:raise ValueError('Oversized NVML response')
        line,extra=data.split(b'\n',1)
        if extra:raise ValueError('Unsolicited NVML response')
        reply=json.loads(line)
        if reply.get('sequence')!=self.sequence:raise ValueError('Stale NVML response')
        if 'error' in reply:raise RuntimeError(reply['error'])
        if time.monotonic()>=deadline:raise TimeoutError('Late NVML response')
        reply['spawn_to_exec_seconds']=spawn_seconds
        return reply

    def close(self, deadline=None):
        with self.state_lock:
            self.closed=True
            process=self.process
        if process is None:return
        self._reap(process,time.monotonic()+.2 if deadline is None else deadline)

    @staticmethod
    def _reap(process, deadline):
        if process.poll() is None:process.kill()
        process.wait(timeout=max(0,deadline-time.monotonic()))
        for stream in (process.stdin,process.stdout):
            if stream is not None:stream.close()


def parent_fence(expected):
    # Linux-only operator: no orphan helper if its owner is killed outright.
    if ctypes.CDLL(None).prctl(1,signal.SIGKILL,0,0,0)!=0 or os.getppid()!=expected:
        raise RuntimeError('NVML helper parent ownership unavailable')


def main():
    parent_fence(int(sys.argv[2]))
    started=time.monotonic();sensor=NVML(json.loads(sys.argv[1]));initialization=time.monotonic()-started
    try:
        for line in sys.stdin:
            if len(line)>16:raise ValueError('Invalid sensor request')
            sequence=int(line);query_started=time.monotonic()
            try:reply=sensor.sample()
            except Exception as exc:reply=dict(error=f'{type(exc).__name__}: {exc}'[:512])
            reply.update(sequence=sequence,helper_query_seconds=time.monotonic()-query_started,
                         initialization_seconds=initialization)
            print(json.dumps(reply,allow_nan=False),flush=True)
    finally:sensor.call(sensor.shutdown)


if __name__=='__main__':main()
