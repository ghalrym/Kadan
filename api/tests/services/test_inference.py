import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from api.routes.v1.images.generations import ImageRequest
from api.routes.v1.chat.completions import CompletionRequest
from api.pydantic_models.inference import ImageProgress
from api.services.inference import InferenceService
from api.services.inference_worker import InferenceJob, InferenceTask, WorkerEvent


class InferenceBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_completion_only_relays_worker_result(self):
        service = InferenceService()
        job = InferenceJob(
            InferenceTask(input={"feature": "image", "prompt": "test"}), False
        )
        service.worker.jobs[job.request.id] = job
        task = asyncio.create_task(service._complete(job))
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        result = {"image": {"images_base64": ["worker-result"]}}
        job.result.set_result(result)
        self.assertEqual(await task, result)
        self.assertNotIn(job.request.id, service.worker.jobs)

    async def test_submission_contains_typed_complete_inputs(self):
        service = InferenceService()

        async def submit(request, **kwargs):
            job = InferenceJob(request, False)
            self.assertEqual(request.input.prompt, "a mountain")
            self.assertEqual(request.input.count, 2)
            self.assertEqual(request.input.seed, 17)
            job.result.set_result({"image": {}})
            return job

        with patch.object(service.worker, "submit", side_effect=submit):
            self.assertEqual(
                await service.submit(
                    ImageRequest(prompt="a mountain", count=2, seed=17), feature="image"
                ),
                {"image": {}},
            )

    async def test_stream_relays_content_and_uses_worker_selected_model(self):
        service = InferenceService()
        body = CompletionRequest(messages=[{"role": "user", "text": "Hello"}])
        job = InferenceJob(
            InferenceTask(
                operation="completion",
                input={
                    "feature": "llm",
                    "messages": [{"role": "user", "text": "Hello"}],
                },
            ),
            True,
        )
        completion = asyncio.create_task(service._complete(job))
        with patch.object(
            service, "_submit", new=AsyncMock(return_value=(job, completion))
        ):
            opening = asyncio.create_task(service.open_chat_stream(body))
            await asyncio.sleep(0)
            self.assertFalse(opening.done())
            job.model = "small"
            job.identified.set()
            stream, model = await opening
            self.assertEqual(model, "small")
            job.events.put_nowait({"content": "Hello back"})
            job.result.set_result({"finish_reason": "stop"})
            self.assertEqual(
                [event async for event in stream],
                [{"content": "Hello back"}, {"finish_reason": "stop"}],
            )

    async def test_immediate_cancel_releases_native_job_before_task_first_step(self):
        service = InferenceService()
        job = InferenceJob(
            InferenceTask(input={"feature": "image", "prompt": "test"}), False
        )
        service.worker.jobs[job.request.id] = job
        task = asyncio.create_task(service._complete(job))
        service.tasks[job.request.id] = task

        async def join(_):
            job.result.set_result(None)

        with patch.object(
            service.worker, "cancel_and_join", side_effect=join
        ) as cleanup:
            await service.cancel(job.request.id)
        cleanup.assert_awaited_once_with(job)
        self.assertTrue(task.cancelled())
        self.assertNotIn(job.request.id, service.worker.jobs)

    async def test_close_tolerates_job_completing_during_another_cancellation(self):
        service = InferenceService()
        head = InferenceJob(
            InferenceTask(input={"feature": "image", "prompt": "test"}), False
        )
        tail = InferenceJob(
            InferenceTask(input={"feature": "image", "prompt": "test"}), False
        )
        service.worker.jobs.update({head.request.id: head, tail.request.id: tail})

        async def join(job):
            tail.result.set_result(None)
            service.worker.jobs.pop(tail.request.id)
            job.result.set_result(None)

        with (
            patch.object(service.worker, "cancel_and_join", side_effect=join),
            patch.object(service.worker, "close", new=AsyncMock()) as close,
        ):
            await service.close()
        close.assert_awaited_once()
        self.assertFalse(service.worker.jobs)


class ImageProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_steps_survive_block_events_and_reset_for_next_image(self):
        service = InferenceService()
        job = InferenceJob(
            InferenceTask(input={"feature": "image", "prompt": "test"}), False
        )
        job.image_progress = ImageProgress(image_index=1, image_count=4, steps_total=20)

        def report(phase, value):
            service._observe(
                job,
                WorkerEvent(
                    id=job.request.id, state="progress", phase=phase, value=value
                ),
            )

        report("image_started", 1)
        report("step_completed", 3)
        report("block", 0)
        self.assertEqual(service.progress["image"]["steps_completed"], 3)
        self.assertEqual(service.progress["image"]["steps_total"], 20)
        report("image_started", 2)
        report("block", 15)
        self.assertEqual(service.progress["image"]["image_index"], 2)
        self.assertEqual(service.progress["image"]["image_count"], 4)
        self.assertEqual(service.progress["image"]["steps_completed"], 0)
        service._observe(job, WorkerEvent(id=job.request.id, state="succeeded"))
        self.assertIsNone(service.progress)
