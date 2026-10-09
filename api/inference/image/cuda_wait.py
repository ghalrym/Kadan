"""Rank-local CUDA 13 wait policy; importing this module never loads CUDA."""
import ctypes
import os
from pathlib import Path

SCHEDULE_MASK = 0x07
BLOCKING_SYNC = 0x04


def loaded_runtime_path(maps):
    paths = set()
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and Path(fields[5]).name.startswith('libcudart.so'):
            paths.add(fields[5])
    if len(paths) != 1:
        raise RuntimeError('Exactly one already-loaded CUDA runtime is required')
    path = paths.pop()
    if not path.startswith('/') or path.endswith(' (deleted)'):
        raise RuntimeError('Invalid loaded CUDA runtime path')
    return path


class Runtime:
    def __init__(self):
        path = loaded_runtime_path(Path('/proc/self/maps').read_text())
        # Never introduce a second runtime or silently load a different version.
        self.library = ctypes.CDLL(path, mode=os.RTLD_LOCAL | os.RTLD_NOLOAD)
        for name, arguments in (
            ('cudaRuntimeGetVersion', [ctypes.POINTER(ctypes.c_int)]),
            ('cudaSetDevice', [ctypes.c_int]),
            ('cudaGetDevice', [ctypes.POINTER(ctypes.c_int)]),
            ('cudaGetDeviceFlags', [ctypes.POINTER(ctypes.c_uint)]),
            ('cudaSetDeviceFlags', [ctypes.c_uint]),
        ):
            function = getattr(self.library, name)
            function.argtypes, function.restype = arguments, ctypes.c_int

    def call(self, name, *arguments):
        status = getattr(self.library, name)(*arguments)
        if status != 0:
            raise RuntimeError(f'{name} failed with CUDA status {status}')

    def query(self, name, value_type=ctypes.c_int):
        value = value_type()
        self.call(name, ctypes.byref(value))
        return value.value

    def version(self):
        return self.query('cudaRuntimeGetVersion')

    def set_device(self, device):
        self.call('cudaSetDevice', device)

    def device(self):
        return self.query('cudaGetDevice')

    def flags(self):
        return self.query('cudaGetDeviceFlags', ctypes.c_uint)

    def set_flags(self, flags):
        self.call('cudaSetDeviceFlags', flags)


def configure_blocking_sync(device, runtime=None):
    """Call after Torch binds this owned rank device, before groups/model work.

    Fail closed on any CUDA error or readback mismatch. The existing controller
    owns process teardown; do not try to reuse a partially configured rank.
    """
    if type(device) is not int or device < 0:
        raise ValueError('An explicit nonnegative rank device is required')
    runtime = Runtime() if runtime is None else runtime
    version = runtime.version()
    if not 13000 <= version < 14000:
        raise RuntimeError('CUDA 13 runtime required for initialized-context flags')
    runtime.set_device(device)
    if runtime.device() != device:
        raise RuntimeError('CUDA rank device binding differs')
    before = runtime.flags()
    desired = (before & ~SCHEDULE_MASK) | BLOCKING_SYNC
    runtime.set_flags(desired)
    after = runtime.flags()
    if runtime.device() != device or after != desired:
        raise RuntimeError('CUDA blocking-wait readback differs')
    return dict(device=device, runtime_version=version, before=before, after=after)
