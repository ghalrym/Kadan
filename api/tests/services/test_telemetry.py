import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch

from api.services.telemetry import memory_meters


class MemoryMeterTests(unittest.TestCase):
    def test_host_memory_does_not_import_torch(self):
        with patch('builtins.open', mock_open(read_data='MemTotal: 2097152 kB\nMemAvailable: 1048576 kB\n')), patch.dict(sys.modules, {'torch': None}):
            samples, errors = memory_meters()
        self.assertEqual(samples, [dict(label='Host RAM', used=1.0, total=2.0)])
        self.assertEqual(errors, ['GPU measurement unavailable until inference dependencies are loaded'])

    def test_loaded_cuda_samples_keep_device_order(self):
        cuda = Mock()
        cuda.device_count.return_value = 2
        cuda.mem_get_info.side_effect = [(1024**3, 2 * 1024**3), (3 * 1024**3, 4 * 1024**3)]
        with patch('builtins.open', side_effect=OSError), patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=cuda)}):
            samples, errors = memory_meters()
        self.assertEqual(samples, [dict(label='GPU 0 · VRAM', used=1.0, total=2.0), dict(label='GPU 1 · VRAM', used=1.0, total=4.0)])
        self.assertEqual(errors, ['Host RAM measurement unavailable'])
        self.assertEqual([call.args for call in cuda.mem_get_info.call_args_list], [(0,), (1,)])

    def test_unavailable_cuda_reports_independent_errors(self):
        cuda = Mock()
        cuda.is_available.return_value = False
        with patch('builtins.open', side_effect=OSError), patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=cuda)}):
            samples, errors = memory_meters()
        self.assertEqual(samples, [])
        self.assertEqual(errors, ['Host RAM measurement unavailable', 'CUDA GPU measurement unavailable'])
        cuda.mem_get_info.assert_not_called()

    def test_failed_device_probe_keeps_earlier_samples(self):
        cuda = Mock()
        cuda.device_count.return_value = 2
        cuda.mem_get_info.side_effect = [(1024**3, 2 * 1024**3), RuntimeError('driver failed')]
        with patch('builtins.open', mock_open(read_data='MemTotal: 2097152 kB\nMemAvailable: 1048576 kB\n')), patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=cuda)}):
            samples, errors = memory_meters()
        self.assertEqual(samples, [dict(label='Host RAM', used=1.0, total=2.0), dict(label='GPU 0 · VRAM', used=1.0, total=2.0)])
        self.assertEqual(errors, ['GPU measurement failed'])
