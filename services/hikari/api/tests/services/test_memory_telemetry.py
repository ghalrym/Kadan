import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, mock_open, patch

from api.services.memory_telemetry import physical_memory

GIB = 1024**3


def worker(name="llm", computer="boot-1", gpus=("GPU-0",), used=(8,)):
    ready = asyncio.Event()
    ready.set()
    return SimpleNamespace(
        id=name,
        ready=ready,
        devices=list(range(1, len(gpus) + 1)),
        computer=SimpleNamespace(
            id=computer,
            name="box",
            gpus={uuid: index + 1 for index, uuid in enumerate(gpus)},
        ),
        control=AsyncMock(
            return_value={
                "machine": {
                    "id": computer,
                    "devices": [
                        {
                            "uuid": uuid,
                            "vram_total": 24 * GIB,
                            "vram_used": value * GIB,
                            # Admission free bytes deliberately disagree.
                            "vram_free": 2 * GIB,
                        }
                        for uuid, value in zip(gpus, used)
                    ],
                }
            }
        ),
    )


class PhysicalMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def sample(self, *sources):
        with patch(
            "api.services.memory_telemetry.open",
            mock_open(read_data="MemTotal: 8192 kB\nMemAvailable: 2048 kB\n"),
        ):
            return await physical_memory({source.id: source for source in sources})

    async def test_each_poll_reads_physical_used_bytes_not_admission_free_bytes(self):
        source = worker()
        first, errors = await self.sample(source)
        self.assertEqual(errors, [])
        self.assertEqual([r.used_bytes for r in first], [6144 * 1024, 8 * GIB])
        source.control.return_value["machine"]["devices"][0]["vram_used"] = GIB
        second, errors = await self.sample(source)
        self.assertEqual(errors, [])
        self.assertEqual(second[1].used_bytes, GIB)
        self.assertEqual(second[1].label, "GPU 0 memory used")
        self.assertEqual(source.control.await_count, 2)

    async def test_two_gpus_are_deduplicated_across_workers_by_computer_and_uuid(self):
        first = worker(gpus=("GPU-0", "GPU-1"), used=(8, 5))
        replica = worker("image", gpus=("GPU-1", "GPU-0"), used=(5, 8))
        replica.computer = first.computer
        replica.devices = [2, 1]
        readings, errors = await self.sample(first, replica)
        self.assertEqual(errors, [])
        self.assertEqual([r.used_bytes for r in readings[1:]], [8 * GIB, 5 * GIB])
        self.assertEqual(
            [r.label for r in readings[1:]],
            ["GPU 0 memory used", "GPU 1 memory used"],
        )

    async def test_disjoint_gpu_visibility_is_combined(self):
        first, second = worker(), worker("image", gpus=("GPU-1",), used=(5,))
        first.computer.gpus["GPU-1"] = 2
        second.computer = first.computer
        second.devices = [2]
        readings, errors = await self.sample(first, second)
        self.assertEqual(errors, [])
        self.assertEqual(len(readings), 3)
        self.assertEqual(readings[2].used_bytes, 5 * GIB)

    async def test_separate_computers_have_distinct_labels(self):
        readings, errors = await self.sample(worker(), worker("remote", "boot-2"))
        self.assertEqual(errors, [])
        self.assertEqual(len(readings), 3)
        self.assertNotEqual(readings[1].label, readings[2].label)

    async def test_failed_replica_does_not_hide_another_current_measurement(self):
        first, second = worker(), worker("image")
        first.control.side_effect = TimeoutError()
        readings, errors = await self.sample(first, second)
        self.assertEqual(errors, [])
        self.assertEqual(len(readings), 2)

    async def test_missing_gpu_does_not_hide_available_gpu_or_host(self):
        first, second = worker(), worker("image", gpus=("GPU-1",))
        first.computer.gpus["GPU-1"] = 2
        second.computer = first.computer
        second.devices = [2]
        second.control.side_effect = RuntimeError("disconnected")
        readings, errors = await self.sample(first, second)
        self.assertEqual(len(readings), 2)
        self.assertEqual(errors, ["Physical GPU memory measurement unavailable"])

    async def test_failed_poll_never_reuses_previous_readings(self):
        source = worker()
        await self.sample(source)
        source.control.side_effect = TimeoutError()
        readings, errors = await self.sample(source)
        self.assertEqual(len(readings), 1)
        self.assertEqual(len(errors), 1)

    async def test_invalid_or_missing_readings_preserve_host(self):
        for devices in (
            [],
            [{"uuid": "GPU-0", "vram_total": 24 * GIB}],
            [{"uuid": "GPU-0", "vram_total": 24 * GIB, "vram_used": -1}],
            [{"uuid": "GPU-0", "vram_total": GIB, "vram_used": 2 * GIB}],
            [{"uuid": "GPU-0", "vram_total": 0, "vram_used": 0}],
            [{"uuid": "GPU-0", "vram_total": GIB, "vram_used": "N/A"}],
            [{"uuid": "unknown", "vram_total": GIB, "vram_used": 0}],
        ):
            with self.subTest(devices=devices):
                source = worker()
                source.control.return_value["machine"]["devices"] = devices
                readings, errors = await self.sample(source)
                self.assertEqual(len(readings), 1)
                self.assertEqual(len(errors), 1)

    async def test_wrong_computer_is_rejected(self):
        source = worker()
        source.control.return_value["machine"]["id"] = "other-host"
        readings, errors = await self.sample(source)
        self.assertEqual(len(readings), 1)
        self.assertEqual(len(errors), 1)

    async def test_no_gpu_or_unready_worker_reports_unavailable(self):
        source = worker()
        source.ready.clear()
        for sources in ((), (source,), (worker(gpus=(), used=()),)):
            readings, errors = await self.sample(*sources)
            self.assertEqual(len(readings), 1)
            self.assertEqual(len(errors), 1)
        source.control.assert_not_awaited()

    async def test_disconnected_worker_cannot_supply_a_reading(self):
        source = worker()
        sources = {source.id: source}

        async def disconnected(command):
            sources.clear()
            return source.control.return_value

        source.control.side_effect = disconnected
        readings, errors = await physical_memory(sources)
        self.assertEqual(len(readings), 1)
        self.assertEqual(len(errors), 1)

    async def test_host_failure_preserves_gpu_readings(self):
        source = worker()
        with patch("api.services.memory_telemetry.open", side_effect=OSError()):
            readings, errors = await physical_memory({source.id: source})
        self.assertEqual(len(readings), 1)
        self.assertEqual(readings[0].used_bytes, 8 * GIB)
        self.assertEqual(errors, ["Host RAM measurement unavailable"])
