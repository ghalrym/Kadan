import unittest
from types import SimpleNamespace
from unittest.mock import patch

from api.routes.v1 import metrics
from api.services.memory_telemetry import MemoryReading


class MemoryDisplayTests(unittest.TestCase):
    def test_fresh_physical_readings_are_separate_from_old_reservations(self):
        worker = SimpleNamespace(memory=SimpleNamespace(used=[1024**3, 400 * 1024**2]),
                                 memory_observed_at='2026-10-11T10:00:00+00:00')
        with patch.object(metrics.inference, 'worker', worker), \
             patch.object(metrics, 'physical_memory', side_effect=[
                 ([MemoryReading(label='GPU 0 memory used', used_bytes=22175 * 1024**2,
                                 total_bytes=24124 * 1024**2)], []),
                 ([], ['Physical GPU memory measurement unavailable'])]):
            first, second = metrics.get_metrics(), metrics.get_metrics()
        self.assertEqual(first.resources[0].used, 22175 / 1024)
        self.assertEqual(first.reservations[1].reserved_gib, 400 / 1024)
        self.assertEqual(first.reservations_observed_at, worker.memory_observed_at)
        self.assertEqual(second.resources, [])
        self.assertEqual(len(second.resource_errors), 1)
