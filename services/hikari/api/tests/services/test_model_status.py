import asyncio
import unittest
from unittest.mock import patch

from api.pydantic_models.inference_task import InferenceTask, ModelContextTask
from api.pydantic_models.model_store import ModelStoreStatus
from api.services.inference import InferenceService
from api.services.inference_errors import InferenceFailure
from api.services.inference_worker import InferenceJob, WorkerEvent


def snapshot():
    return ModelStoreStatus.model_validate(
        {
            "models": [
                {
                    "id": "small",
                    "repo_id": "test/model",
                    "revision": "pinned",
                    "license": "MIT",
                    "estimated_bytes": 20,
                    "kind": "llm",
                    "display_name": "Small",
                    "license_url": None,
                    "license_notice": None,
                    "inference_available": True,
                    "status": "not_downloaded",
                    "downloaded_bytes": 0,
                    "total_bytes": 0,
                    "error": None,
                    "context_limit": 65536,
                    "architecture_context_limit": None,
                }
            ],
            "selected_model_id": None,
        }
    )


class ModelStatusTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = InferenceService()
        self.service.worker.models = snapshot()

        async def submit(request, **kwargs):
            job = InferenceJob(request, False)
            self.service.worker.jobs[request.id] = job
            return job

        async def cancel(job_id):
            job = self.service.worker.jobs[job_id]
            self.service._observe(job, WorkerEvent(state="cancelled"))
            job.result.set_result({})

        self.submit = patch.object(self.service.worker, "submit", side_effect=submit)
        self.cancel = patch.object(self.service.worker, "cancel", side_effect=cancel)
        self.submit.start()
        self.cancel.start()
        self.addCleanup(self.submit.stop)
        self.addCleanup(self.cancel.stop)

    async def test_download_uses_existing_queue_and_snapshot(self):
        await self.service.download("small", True)
        job = next(iter(self.service.worker.jobs.values()))
        self.assertEqual(job.request.operation, "download")
        self.assertTrue(job.request.input.license_acknowledged)
        self.service._observe(
            job, WorkerEvent(state="progress", phase="download", value=10, total=20)
        )
        status = self.service.models_status().models[0]
        self.assertEqual(
            (status.status, status.downloaded_bytes, status.total_bytes),
            ("downloading", 10, 20),
        )
        self.assertEqual(self.service.worker.models.models[0].status, "not_downloaded")
        self.service.worker.models.models[0].status = "complete"
        self.service._observe(job, WorkerEvent(state="succeeded"))
        job.result.set_result({})
        await asyncio.gather(*self.service.tasks.values())
        self.assertEqual(self.service.models_status().models[0].status, "complete")
        self.assertFalse(self.service.worker.jobs)

    async def test_duplicate_and_cancel_use_outstanding_job(self):
        await self.service.download("small", False)
        with self.assertRaises(InferenceFailure):
            await self.service.download("small", False)
        await self.service.cancel_download("small")
        await asyncio.gather(*self.service.tasks.values())
        self.assertEqual(self.service.models_status().models[0].status, "cancelled")
        with self.assertRaises(InferenceFailure):
            await self.service.cancel_download("small")

    async def test_duplicate_download_is_rejected_while_worker_admission_waits(self):
        service = InferenceService()
        service.worker.models = snapshot()
        entered, release = asyncio.Event(), asyncio.Event()

        async def wait(request):
            entered.set()
            await release.wait()

        async def accept(job):
            job.accepted.set_result(None)

        with (
            patch.object(
                service.worker, "_wait_for_worker", side_effect=wait
            ) as waiting,
            patch.object(service.worker, "_send_job", side_effect=accept) as sending,
        ):
            first = asyncio.create_task(service.download("small", True))
            await entered.wait()
            observation = service.downloads["small"]
            second = asyncio.create_task(service.download("small", True))
            try:
                await asyncio.sleep(0)
                self.assertTrue(second.done())
                with self.assertRaises(InferenceFailure) as failure:
                    await second
                self.assertEqual(failure.exception.status_code, 409)
                self.assertIs(service.downloads["small"], observation)
                waiting.assert_awaited_once()
                self.assertFalse(service.worker.jobs)
                release.set()
                await first
                sending.assert_awaited_once()
                self.assertEqual(len(service.worker.jobs), 1)
            finally:
                release.set()
                await asyncio.gather(first, second, return_exceptions=True)
                for job in tuple(service.worker.jobs.values()):
                    service._observe(job, WorkerEvent(state="succeeded"))
                    job.result.set_result({})
                await asyncio.gather(*service.tasks.values())

    async def test_another_model_can_queue_while_first_download_waits_for_admission(
        self,
    ):
        service = InferenceService()
        service.worker.models = snapshot()
        service.worker.models.models.append(
            service.worker.models.models[0].model_copy(update={"id": "other"})
        )
        entered, release = asyncio.Event(), asyncio.Event()

        async def wait(request):
            if request.input.model_id == "small":
                entered.set()
                await release.wait()

        async def accept(job):
            job.accepted.set_result(None)

        with (
            patch.object(service.worker, "_wait_for_worker", side_effect=wait),
            patch.object(service.worker, "_send_job", side_effect=accept),
        ):
            first = asyncio.create_task(service.download("small", True))
            await entered.wait()
            try:
                await service.download("other", True)
                self.assertFalse(first.done())
                self.assertEqual(
                    [
                        job.request.input.model_id
                        for job in service.worker.jobs.values()
                    ],
                    ["other"],
                )
                release.set()
                await first
                self.assertEqual(len(service.worker.jobs), 2)
            finally:
                release.set()
                await asyncio.gather(first, return_exceptions=True)
                for job in tuple(service.worker.jobs.values()):
                    service._observe(job, WorkerEvent(state="succeeded"))
                    job.result.set_result({})
                await asyncio.gather(*service.tasks.values())

    async def test_failed_or_cancelled_admission_allows_retry(self):
        for error in (
            InferenceFailure("Worker unavailable.", 503),
            asyncio.CancelledError(),
        ):
            with self.subTest(error=type(error).__name__):
                self.service.worker.submit.side_effect = error
                with self.assertRaises(type(error)):
                    await self.service.download("small", True)
                self.assertEqual(
                    self.service.models_status().models[0].status, "failed"
                )

                async def submit(request, **kwargs):
                    job = InferenceJob(request, False)
                    self.service.worker.jobs[request.id] = job
                    return job

                self.service.worker.submit.side_effect = submit
                await self.service.download("small", True)
                job = next(iter(self.service.worker.jobs.values()))
                self.service._observe(job, WorkerEvent(state="succeeded"))
                job.result.set_result({})
                await asyncio.gather(*self.service.tasks.values())

    async def test_configuration_is_typed_and_waits_for_worker(self):
        task = asyncio.create_task(
            self.service.submit(
                ModelContextTask(model_id="small", context_limit=None),
                feature="model_context",
                operation="configure",
            )
        )
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        job = next(iter(self.service.worker.jobs.values()))
        self.assertIsNone(job.request.model_dump()["input"]["context_limit"])
        self.service.worker.models.models[0].context_limit = None
        job.result.set_result({})
        await task
        self.assertIsNone(self.service.models_status().models[0].context_limit)

    async def test_missing_worker_status_is_unavailable(self):
        self.service.worker.models = None
        with self.assertRaises(InferenceFailure):
            self.service.models_status()

    async def test_worker_failure_is_presented_without_local_file_validation(self):
        await self.service.download("small", False)
        job = next(iter(self.service.worker.jobs.values()))
        self.service._observe(
            job, WorkerEvent(state="failed", error="Integrity failure")
        )
        job.result.set_result({})
        await asyncio.gather(*self.service.tasks.values())
        state = self.service.models_status().models[0]
        self.assertEqual((state.status, state.error), ("failed", "Integrity failure"))

    async def test_lost_download_completion_allows_incomplete_model_to_resume(self):
        for active in ("downloading", "cancelling"):
            with self.subTest(status=active):
                await self.service.download("small", True)
                job = next(iter(self.service.worker.jobs.values()))
                self.service.downloads["small"].status = active
                job.result.set_exception(InferenceFailure("Worker disconnected.", 503))
                with self.assertRaisesRegex(InferenceFailure, "Worker disconnected"):
                    await asyncio.gather(*self.service.tasks.values())
                self.assertFalse(self.service.worker.jobs)

                # A replacement worker reports the retained, incomplete store.
                self.service.worker.models = snapshot()
                state = self.service.models_status().models[0]
                self.assertEqual(
                    (state.status, state.error), ("failed", "Worker disconnected.")
                )
                await self.service.download("small", True)
                retry = next(iter(self.service.worker.jobs.values()))
                self.assertNotEqual(retry.request.id, job.request.id)
                self.assertEqual(
                    self.service.models_status().models[0].status, "downloading"
                )
                self.service.worker.models.models[0].status = "complete"
                self.service._observe(retry, WorkerEvent(state="succeeded"))
                retry.result.set_result({})
                await asyncio.gather(*self.service.tasks.values())
                self.assertEqual(
                    self.service.models_status().models[0].status, "complete"
                )

    async def test_completion_failure_preserves_observed_terminal_download_outcomes(
        self,
    ):
        for terminal in ("failed", "cancelled", "succeeded"):
            with self.subTest(status=terminal):
                await self.service.download("small", True)
                job = next(iter(self.service.worker.jobs.values()))
                observed_error = "" if terminal == "succeeded" else "Observed outcome"
                self.service._observe(
                    job, WorkerEvent(state=terminal, error=observed_error)
                )
                job.result.set_exception(InferenceFailure("Worker disconnected.", 503))
                with self.assertRaises(InferenceFailure):
                    await asyncio.gather(*self.service.tasks.values())
                state = self.service.models_status().models[0]
                self.assertEqual(
                    (state.status, state.error),
                    (
                        "complete" if terminal == "succeeded" else terminal,
                        observed_error or None,
                    ),
                )

    async def test_completed_worker_snapshot_survives_download_completion_failure(self):
        await self.service.download("small", True)
        job = next(iter(self.service.worker.jobs.values()))
        self.service.worker.models.models[0].status = "complete"
        job.result.set_exception(InferenceFailure("Worker disconnected.", 503))
        with self.assertRaises(InferenceFailure):
            await asyncio.gather(*self.service.tasks.values())
        state = self.service.models_status().models[0]
        self.assertEqual((state.status, state.error), ("complete", None))

    async def test_late_download_failure_does_not_mark_a_new_retry_failed(self):
        await self.service.download("small", True)
        prior = next(iter(self.service.worker.jobs.values()))
        completion = self.service.tasks[prior.request.id]
        prior.result.set_exception(InferenceFailure("Worker disconnected.", 503))
        await self.service.download("small", True)
        retry = next(
            job for job in self.service.worker.jobs.values() if job is not prior
        )
        with self.assertRaises(InferenceFailure):
            await completion
        state = self.service.models_status().models[0]
        self.assertEqual((state.status, state.error), ("downloading", None))
        self.service.worker.models.models[0].status = "complete"
        self.service._observe(retry, WorkerEvent(state="succeeded"))
        retry.result.set_result({})
        await asyncio.gather(*self.service.tasks.values())


class SettingsTaskTests(unittest.TestCase):
    def test_all_settings_tasks_round_trip(self):
        for body in (
            {"feature": "model_context", "model_id": "small", "context_limit": None},
            {"feature": "whisper_selection", "model": "tiny.en"},
            {"feature": "whisper_formatting", "enabled": False},
        ):
            task = InferenceTask(operation="configure", input=body)
            self.assertEqual(
                InferenceTask.model_validate_json(task.model_dump_json()), task
            )
