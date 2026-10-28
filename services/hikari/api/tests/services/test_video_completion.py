"""Video completion with in-memory jobs and mocked thread/file boundaries."""

import asyncio
import base64
import unittest
from unittest.mock import AsyncMock, Mock, patch

from api.pydantic_models.inference_task import InferenceTask
from api.services.inference import InferenceService
from api.services.inference_errors import InferenceFailure
from api.services.inference_worker import InferenceJob, WorkerEvent
from api.services.video_jobs import VideoJobs


class VideoCompletionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = InferenceService()
        self.videos = VideoJobs()
        self.store = Mock()
        self.thread = AsyncMock(side_effect=lambda function, *args: function(*args))
        for replacement in (
            patch("api.services.inference.video_jobs", self.videos),
            patch.object(self.videos, "store", self.store),
            patch("api.services.inference.asyncio.to_thread", self.thread),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)
        self.job = self.video_job()

    def video_job(self):
        job = InferenceJob(
            InferenceTask(
                input={
                    "feature": "video",
                    "prompt": "Sea",
                    "duration": 5,
                    "resolution": "480p",
                }
            ),
            False,
        )
        self.videos.prepare(job.request.id, job.request.input)
        self.service.worker.jobs[job.request.id] = job
        return job

    def result(self):
        return {
            "status": "Done",
            "progress": 100,
            "video": base64.b64encode(b"mp4").decode(),
        }

    async def test_worker_result_failure_without_observer_marks_video_failed(self):
        self.videos.update(self.job.request.id, status="Rendering")
        self.job.result.set_exception(InferenceFailure("Worker disconnected.", 503))
        with self.assertRaises(InferenceFailure):
            await self.service._complete(self.job)
        video = self.videos.get(self.job.request.id)
        self.assertEqual(
            (video.status, video.error), ("Failed", "Worker disconnected.")
        )
        self.assertNotIn(self.job.request.id, self.service.worker.jobs)
        self.assertIsNone(video.output_url)

    async def test_artifact_store_failure_marks_video_failed_without_output(self):
        self.store.side_effect = OSError("Disk full")
        self.job.result.set_result(self.result())
        with self.assertRaisesRegex(OSError, "Disk full"):
            await self.service._complete(self.job)
        video = self.videos.get(self.job.request.id)
        self.assertEqual((video.status, video.error), ("Failed", "Disk full"))
        self.assertIsNone(video.output_url)
        with self.assertRaises(KeyError):
            self.videos.content(video.id)

    async def test_cancelled_worker_result_without_observer_remains_cancelled(self):
        self.job.result.set_exception(InferenceFailure("Inference cancelled.", 499))
        with self.assertRaises(InferenceFailure):
            await self.service._complete(self.job)
        self.assertEqual(self.videos.get(self.job.request.id).status, "Cancelled")

    async def test_confirmed_worker_cancellation_is_not_replaced_by_failure(self):
        self.service._observe(self.job, WorkerEvent(state="cancelled"))
        self.job.result.set_exception(InferenceFailure("Inference cancelled.", 499))
        with self.assertRaises(InferenceFailure):
            await self.service._complete(self.job)
        self.assertEqual(self.videos.get(self.job.request.id).status, "Cancelled")

    async def test_cancellation_during_artifact_delivery_finishes_video_state(self):
        entered = asyncio.Event()

        async def delivery(function, *args):
            entered.set()
            await asyncio.Event().wait()

        self.thread.side_effect = delivery
        self.job.result.set_result(self.result())
        task = asyncio.create_task(self.service._complete(self.job))
        self.service.tasks[self.job.request.id] = task
        await entered.wait()
        await self.service.cancel(self.job.request.id)
        video = self.videos.get(self.job.request.id)
        self.assertTrue(task.cancelled())
        self.assertEqual(video.status, "Cancelled")
        self.assertIsNone(video.output_url)
        self.assertNotIn(self.job.request.id, self.service.worker.jobs)
        self.store.assert_not_called()

    async def test_prestart_cancellation_updates_video_after_fallback_cleanup(self):
        task = asyncio.create_task(self.service._complete(self.job))
        self.service.tasks[self.job.request.id] = task

        async def cleanup(job):
            self.assertEqual(self.videos.get(job.request.id).status, "Queued")
            job.result.set_exception(InferenceFailure("Inference cancelled.", 499))

        with patch.object(
            self.service.worker, "cancel_and_join", side_effect=cleanup
        ) as joined:
            await self.service.cancel(self.job.request.id)
        joined.assert_awaited_once_with(self.job)
        self.assertTrue(task.cancelled())
        self.assertEqual(self.videos.get(self.job.request.id).status, "Cancelled")

    async def test_success_keeps_saved_output_and_original_video_id(self):
        unrelated = self.video_job()
        self.job.result.set_result(self.result())
        await self.service._complete(self.job)
        video = self.videos.get(self.job.request.id)
        self.assertEqual((video.id, video.status), (self.job.request.id, "Done"))
        self.assertEqual(video.output_url, f"/v1/videos/{video.id}/content")
        self.assertEqual(self.videos.get(unrelated.request.id).status, "Queued")
        self.store.assert_called_once_with(video.id, b"mp4")

    async def test_late_cancellation_preserves_confirmed_terminal_outcomes(self):
        for status in ("Done", "Failed", "Cancelled"):
            with self.subTest(status=status):
                job = self.video_job()
                self.videos.update(
                    job.request.id,
                    status=status,
                    error="Original error" if status == "Failed" else None,
                    output_url="/saved.mp4" if status == "Done" else None,
                )
                original = self.videos.get(job.request.id)
                job.result.set_result({})
                task = asyncio.create_task(self.service._complete(job))
                self.service.tasks[job.request.id] = task
                await self.service.cancel(job.request.id)
                self.assertEqual(self.videos.get(job.request.id), original)
