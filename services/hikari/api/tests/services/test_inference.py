import base64
import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from api.pydantic_models.inference import ImageProgress
from api.pydantic_models.model_store import ModelStoreStatus
from api.routes.v1.chat.completions import CompletionRequest
from api.routes.v1.images.generations import ImageRequest
from api.routes.v1.videos.generations import VideoGenerationRequest
from api.services.video_jobs import VideoJobs
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

    async def test_video_bytes_arrive_inline_and_are_stored(self):
        service = InferenceService()
        job = InferenceJob(
            InferenceTask(input={"feature": "video", "prompt": "sea"}), False
        )
        service.worker.jobs[job.request.id] = job
        job.result.set_result(
            {
                "status": "Done",
                "progress": 100,
                "video": base64.b64encode(b"mp4").decode(),
            }
        )
        with (
            patch("api.services.inference.video_jobs.store") as store,
            patch("api.services.inference.video_jobs.update") as update,
        ):
            await service._complete(job)
        store.assert_called_once_with(job.request.id, b"mp4")
        update.assert_called_once_with(
            job.request.id,
            status="Done",
            progress=100,
            output_url=f"/v1/videos/{job.request.id}/content",
        )

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

    async def test_video_omissions_use_fixed_defaults_in_job_and_worker(self):
        service = InferenceService()
        service.worker.models = ModelStoreStatus(models=[])

        async def submit(request, **kwargs):
            self.assertEqual(request.input.duration, 5)
            self.assertEqual(request.input.resolution, "480p")
            job = InferenceJob(request, False)
            job.result.set_result({})
            return job

        with (
            patch.object(service.worker, "submit", side_effect=submit),
            patch("api.services.inference.video_jobs", VideoJobs()),
        ):
            job = await service.submit(
                VideoGenerationRequest(prompt="Scene"), feature="video"
            )
            await asyncio.gather(*service.tasks.values())
        self.assertEqual((job.duration, job.resolution), ("5s", "480p"))

    async def test_video_defaults_preserve_each_explicit_clip_setting(self):
        service = InferenceService()
        service.worker.models = ModelStoreStatus(models=[])
        for fields, expected in (
            ({"seconds": 15}, (15, "480p")),
            ({"resolution": "768p"}, (5, "768p")),
            ({"seconds": 1, "resolution": "768p"}, (1, "768p")),
        ):

            async def submit(request, **kwargs):
                self.assertEqual(
                    (request.input.duration, request.input.resolution), expected
                )
                job = InferenceJob(request, False)
                job.result.set_result({})
                return job

            with (
                self.subTest(fields=fields),
                patch.object(service.worker, "submit", side_effect=submit),
                patch("api.services.inference.video_jobs", VideoJobs()),
            ):
                await service.submit(
                    VideoGenerationRequest(prompt="Scene", **fields), feature="video"
                )
                await asyncio.gather(*service.tasks.values())

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

    async def test_request_cancel_before_completion_starts_releases_native_job(self):
        service = InferenceService()

        async def submit(request, **kwargs):
            job = InferenceJob(request, False)
            service.worker.jobs[request.id] = job
            asyncio.current_task().cancel()
            return job

        async def join(job):
            job.result.set_result(None)

        with (
            patch.object(service.worker, "submit", side_effect=submit),
            patch.object(
                service.worker, "cancel_and_join", side_effect=join
            ) as cleanup,
        ):
            request = asyncio.create_task(
                service.submit(ImageRequest(prompt="test"), feature="image")
            )
            with self.assertRaises(asyncio.CancelledError):
                await request
        cleanup.assert_awaited_once()
        self.assertFalse(service.worker.jobs)
        self.assertFalse(service.tasks)

    async def test_stream_close_before_completion_starts_releases_native_job(self):
        service = InferenceService()

        async def submit(request, **kwargs):
            job = InferenceJob(request, True)
            service.worker.jobs[request.id] = job
            job.model = "small"
            job.identified.set()
            return job

        async def join(job):
            job.result.set_result(None)

        with (
            patch.object(service.worker, "submit", side_effect=submit),
            patch.object(
                service.worker, "cancel_and_join", side_effect=join
            ) as cleanup,
        ):
            stream, _ = await service.open_chat_stream(
                CompletionRequest(messages=[{"role": "user", "text": "Hello"}])
            )
            await stream.aclose()
            await stream.aclose()
        cleanup.assert_awaited_once()
        self.assertFalse(service.worker.jobs)
        self.assertFalse(service.tasks)

    async def test_repeated_cancel_waits_for_prestart_cleanup_and_removes_job(self):
        service = InferenceService()
        job = InferenceJob(
            InferenceTask(input={"feature": "image", "prompt": "test"}), False
        )
        service.worker.jobs[job.request.id] = job
        service.tasks[job.request.id] = asyncio.create_task(service._complete(job))
        service.tasks[job.request.id].cancel()
        entered, released = asyncio.Event(), asyncio.Event()

        async def join(_):
            entered.set()
            await released.wait()
            job.result.set_result(None)

        with patch.object(
            service.worker, "cancel_and_join", side_effect=join
        ) as cleanup:
            cancellation = asyncio.create_task(service.cancel(job.request.id))
            await entered.wait()
            for _ in range(2):
                cancellation.cancel()
                await asyncio.sleep(0)
                self.assertFalse(cancellation.done())
                self.assertIs(service.worker.jobs[job.request.id], job)
                self.assertFalse(job.result.done())
            released.set()
            with self.assertRaises(asyncio.CancelledError):
                await cancellation
            self.assertNotIn(job.request.id, service.worker.jobs)
            await service.cancel(job.request.id)
        cleanup.assert_awaited_once_with(job)
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
    async def test_prepared_image_uses_worker_step_total(self):
        for model, requested_steps, prepared_steps in (
            (None, 50, 8),
            ("qwen-image-2.1-turbo", 20, 8),
        ):
            with self.subTest(model=model):
                service = InferenceService()
                job = InferenceJob(
                    InferenceTask(
                        input={
                            "feature": "image",
                            "prompt": "test",
                            "model_id": model,
                            "steps": requested_steps,
                        }
                    ),
                    False,
                )
                service._observe(
                    job,
                    WorkerEvent(
                        id=job.request.id,
                        state="progress",
                        phase="prepared",
                        total=prepared_steps,
                    ),
                )
                self.assertEqual(service.progress.image.steps_total, prepared_steps)
                service._observe(
                    job,
                    WorkerEvent(
                        id=job.request.id,
                        state="progress",
                        phase="step_completed",
                        value=1,
                    ),
                )
                self.assertEqual(service.progress.image.steps_total, prepared_steps)
                self.assertEqual(service.progress.image.steps_completed, 1)

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
        self.assertEqual(service.progress.image.steps_completed, 3)
        self.assertEqual(service.progress.image.steps_total, 20)
        report("image_started", 2)
        report("block", 15)
        self.assertEqual(service.progress.image.image_index, 2)
        self.assertEqual(service.progress.image.image_count, 4)
        self.assertEqual(service.progress.image.steps_completed, 0)
        service._observe(job, WorkerEvent(id=job.request.id, state="succeeded"))
        self.assertIsNone(service.progress)

    async def test_phase_survives_transient_stages_before_first_poll(self):
        service = InferenceService()
        job = InferenceJob(
            InferenceTask(input={"feature": "image", "prompt": "test", "count": 1}),
            False,
        )
        for phase, value in [
            ("image_started", 1),
            ("conditioning", 0),
            ("block_started", 3),
            ("qkv", 3),
        ]:
            service._observe(
                job,
                WorkerEvent(
                    id=job.request.id, state="progress", phase=phase, value=value
                ),
            )
        self.assertEqual(job.progress.image.phase, "denoise")
        self.assertEqual(job.progress.image.steps_completed, 0)
        first = job.progress
        service._observe(
            job,
            WorkerEvent(
                id=job.request.id, state="progress", phase="step_completed", value=1
            ),
        )
        service._observe(
            job,
            WorkerEvent(
                id=job.request.id,
                state="progress",
                phase="gpu_0_weight_planned_bytes",
                value=99,
            ),
        )
        self.assertEqual(job.progress.image.phase, "denoise")
        self.assertEqual(job.progress.image.steps_completed, 1)
        self.assertEqual(first.image.steps_completed, 0)
        service._observe(
            job,
            WorkerEvent(
                id=job.request.id, state="progress", phase="vae_up_block", value=1
            ),
        )
        service._observe(
            job,
            WorkerEvent(
                id=job.request.id,
                state="progress",
                phase="gpu_0_weight_planned_bytes",
                value=99,
            ),
        )
        self.assertEqual(job.progress.image.phase, "decode")
        service._observe(
            job,
            WorkerEvent(
                id=job.request.id, state="progress", phase="image_started", value=2
            ),
        )
        self.assertEqual(job.progress.image.phase, "prompt")
        self.assertEqual(job.progress.image.steps_completed, 0)

    async def test_queued_and_terminal_jobs_do_not_replace_another_active_job(self):
        for terminal in ("succeeded", "failed", "cancelled", "rejected"):
            service = InferenceService()
            active = InferenceJob(
                InferenceTask(input={"feature": "image", "prompt": "a"}), False
            )
            queued = InferenceJob(
                InferenceTask(input={"feature": "image", "prompt": "b"}), False
            )
            service._observe(active, WorkerEvent(id=active.request.id, state="running"))
            progress = service.progress
            service._observe(queued, WorkerEvent(id=queued.request.id, state="queued"))
            self.assertIs(service.progress, progress)
            self.assertEqual(queued.progress.image.phase, "queued")
            service._observe(queued, WorkerEvent(id=queued.request.id, state=terminal))
            self.assertIsNone(queued.progress)
            self.assertIs(service.progress, progress)
            service._observe(active, WorkerEvent(id=active.request.id, state=terminal))
            self.assertIsNone(service.progress)
            self.assertIsNone(active.progress)

    async def test_client_request_id_is_used_by_the_worker(self):
        service = InferenceService()

        async def submit(request, **kwargs):
            self.assertEqual(request.id, "a" * 32)
            job = InferenceJob(request, False)
            job.result.set_result({"image": {}})
            return job

        with patch.object(service.worker, "submit", side_effect=submit):
            await service.submit(
                ImageRequest(prompt="test"), feature="image", job_id="a" * 32
            )


class UnloadTests(unittest.IsolatedAsyncioTestCase):
    def chat_job(self, service):
        job = InferenceJob(InferenceTask(input={"feature": "llm"}), False)
        service.worker.jobs[job.request.id] = job
        return job

    async def unload(self, service, **kwargs):
        async def submit(request, **_):
            job = InferenceJob(request, False)
            job.result.set_result({})
            return job

        with (
            patch.object(service.worker, "submit", side_effect=submit) as worker_submit,
            patch.object(service, "cancel", new=AsyncMock()) as cancel,
        ):
            await service.unload(**kwargs)
        worker_submit.assert_not_awaited()
        return cancel

    async def test_unload_returns_the_language_model_budget(self):
        service = InferenceService()
        with patch.object(service.worker, "offload", new=AsyncMock()) as offload:
            await self.unload(service)
        offload.assert_awaited_once_with("lfm3_llm")

    async def test_unload_cancels_running_chats_by_default(self):
        service = InferenceService()
        job = self.chat_job(service)
        cancel = await self.unload(service)
        cancel.assert_awaited_once_with(job.request.id)
