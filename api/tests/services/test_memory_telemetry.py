import subprocess
import unittest
from unittest.mock import Mock, mock_open, patch

from api.services.memory_telemetry import physical_memory


class PhysicalMemoryTests(unittest.TestCase):
    def sample(self, result):
        with patch('api.services.memory_telemetry.open', mock_open(
                read_data='MemTotal: 8192 kB\nMemAvailable: 2048 kB\n')), \
             patch('api.services.memory_telemetry.subprocess.run', return_value=result):
            return physical_memory()

    def test_each_poll_reads_current_physical_usage_for_each_gpu(self):
        first, errors = self.sample(Mock(stdout='0, 20000, 24576\n1, 21000, 24576\n'))
        self.assertEqual(errors, [])
        self.assertEqual([r.used_bytes for r in first],
                         [6144 * 1024, 20000 * 1024**2, 21000 * 1024**2])
        second, _ = self.sample(Mock(stdout='0, 800, 24576\n1, 900, 24576\n'))
        self.assertEqual(second[1].used_bytes, 800 * 1024**2)
        self.assertEqual(second[1].label, 'GPU 0 memory used')

    def test_unavailable_or_invalid_gpu_probe_preserves_host_reading(self):
        for failure in (FileNotFoundError(), subprocess.TimeoutExpired('nvidia-smi', 1)):
            with self.subTest(failure=failure), \
                 patch('api.services.memory_telemetry.open', mock_open(
                     read_data='MemTotal: 8192 kB\nMemAvailable: 2048 kB\n')), \
                 patch('api.services.memory_telemetry.subprocess.run', side_effect=failure):
                readings, errors = physical_memory()
                self.assertEqual(len(readings), 1)
                self.assertIn('Physical GPU memory measurement unavailable', errors)
        for output in ('', '0, N/A, 24576', '0, 25000, 24576', '0, -1, 24576'):
            with self.subTest(output=output):
                readings, errors = self.sample(Mock(stdout=output))
                self.assertEqual(len(readings), 1)
                self.assertEqual(len(errors), 1)
