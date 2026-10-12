import unittest
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

from api.routes.v1 import metrics
from api.services.inference import InferenceService
from api.services.inference_worker import InferenceJob, InferenceTask, WorkerEvent
from api.services.memory_telemetry import MemoryReading


class MemoryDisplayTests(unittest.TestCase):
    def test_fresh_physical_readings_are_separate_from_old_reservations(self):
        worker = SimpleNamespace(
            memory=SimpleNamespace(used=[1024**3, 400 * 1024**2]),
            memory_observed_at="2026-10-11T10:00:00+00:00",
        )
        with (
            patch.object(metrics.inference, "worker", worker),
            patch.object(
                metrics,
                "physical_memory",
                side_effect=[
                    (
                        [
                            MemoryReading(
                                label="GPU 0 memory used",
                                used_bytes=22175 * 1024**2,
                                total_bytes=24124 * 1024**2,
                            )
                        ],
                        [],
                    ),
                    ([], ["Physical GPU memory measurement unavailable"]),
                ],
            ),
        ):
            first, second = metrics.get_metrics(), metrics.get_metrics()
        self.assertEqual(first.resources[0].used, 22175 / 1024)
        self.assertEqual(first.reservations[1].reserved_gib, 400 / 1024)
        self.assertEqual(first.reservations_observed_at, worker.memory_observed_at)
        self.assertEqual(second.resources, [])
        self.assertEqual(len(second.resource_errors), 1)


class ImageRequestProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_scoped_metrics_never_use_another_job(self):
        service = InferenceService()
        image = InferenceJob(
            InferenceTask(input={"feature": "image", "prompt": "test"}), False
        )
        other = InferenceJob(InferenceTask(input={"feature": "llm"}), False)
        service.worker.jobs.update({image.request.id: image, other.request.id: other})
        service._observe(image, WorkerEvent(id=image.request.id, state="running"))
        image_progress = image.progress
        service._observe(other, WorkerEvent(id=other.request.id, state="running"))
        with (
            patch.object(metrics, "inference", service),
            patch.object(metrics, "physical_memory", return_value=([], [])),
        ):
            self.assertEqual(
                metrics.get_metrics(UUID(image.request.id)).inference_progress,
                image_progress,
            )
            self.assertEqual(metrics.get_metrics().inference_progress, other.progress)
            self.assertIsNone(
                metrics.get_metrics(UUID(other.request.id)).inference_progress
            )
            self.assertIsNone(metrics.get_metrics(UUID("f" * 32)).inference_progress)
            image.result.set_result({})
            self.assertIsNone(
                metrics.get_metrics(UUID(image.request.id)).inference_progress
            )
            service.worker.jobs.pop(image.request.id)
            self.assertIsNone(
                metrics.get_metrics(UUID(image.request.id)).inference_progress
            )
