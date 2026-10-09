import unittest
from unittest.mock import patch

from api.inference.image.cuda_wait import configure_blocking_sync, loaded_runtime_path, Runtime


class FakeRuntime:
    def __init__(self, flags=0x38, version=13001):
        self.current, self.value, self.runtime_version = 0, flags, version
        self.calls = []

    def version(self): return self.runtime_version
    def set_device(self, device):
        self.calls.append(('device', device)); self.current = device
    def device(self): return self.current
    def flags(self):
        self.calls.append(('get', self.current)); return self.value
    def set_flags(self, flags):
        self.calls.append(('flags', self.current, flags)); self.value = flags


class CudaWaitTests(unittest.TestCase):
    def test_binds_each_rank_before_flags_and_preserves_other_bits(self):
        for device in (0, 1):
            for schedule in (0, 1, 2, 4):
                runtime = FakeRuntime(0x38 | schedule)
                self.assertEqual(configure_blocking_sync(device, runtime),
                    dict(device=device, runtime_version=13001, before=0x38 | schedule, after=0x3c))
                self.assertEqual(runtime.calls,
                    [('device', device), ('get', device), ('flags', device, 0x3c), ('get', device)])

    def test_binding_failure_never_changes_flags(self):
        runtime = FakeRuntime()
        runtime.device = lambda: 0
        with self.assertRaisesRegex(RuntimeError, 'binding differs'):
            configure_blocking_sync(1, runtime)
        self.assertEqual(runtime.calls, [('device', 1)])

    def test_readback_and_unrelated_flag_changes_fail_closed(self):
        for readback in (0x38, 0x04):
            runtime = FakeRuntime()
            runtime.set_flags = lambda flags: setattr(runtime, 'value', readback)
            with self.assertRaisesRegex(RuntimeError, 'readback differs'):
                configure_blocking_sync(1, runtime)

    def test_unsupported_version_and_invalid_device_do_not_mutate(self):
        for version in (12090, 14000):
            runtime = FakeRuntime(version=version)
            with self.assertRaisesRegex(RuntimeError, 'CUDA 13'):
                configure_blocking_sync(1, runtime)
            self.assertEqual(runtime.calls, [])
        for device in (-1, True, '1'):
            with patch('api.inference.image.cuda_wait.Runtime') as constructor:
                with self.assertRaises(ValueError): configure_blocking_sync(device)
                constructor.assert_not_called()

    def test_runtime_failures_propagate_without_fallback(self):
        for method in ('version', 'set_device', 'device', 'flags', 'set_flags'):
            runtime = FakeRuntime()
            with patch.object(runtime, method, side_effect=RuntimeError('runtime_error')):
                with self.assertRaisesRegex(RuntimeError, 'runtime_error'):
                    configure_blocking_sync(1, runtime)

    def test_loader_requires_one_existing_runtime(self):
        line = '123-456 r-xp 0 00:00 1 /installed/nvidia/cu13/lib/libcudart.so.13'
        self.assertEqual(loaded_runtime_path(line+'\n'+line), '/installed/nvidia/cu13/lib/libcudart.so.13')
        for maps in ('', line+'\n'+line.replace('cu13', 'other'), line+' (deleted)'):
            with self.assertRaises(RuntimeError): loaded_runtime_path(maps)

    def test_ctypes_error_status_is_not_ignored(self):
        runtime = Runtime.__new__(Runtime)
        class Library:
            def cudaSetDeviceFlags(self, flags): return 1
        runtime.library = Library()
        with self.assertRaisesRegex(RuntimeError, 'CUDA status 1'):
            runtime.set_flags(4)


if __name__ == '__main__':
    unittest.main()
