import asyncio
import unittest
from unittest.mock import patch

from api.pydantic_models.inference_task import InferenceTask, ModelContextTask
from api.pydantic_models.model_store import HardwareTier, ModelStoreStatus
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

    async def test_tier_download_queues_only_incomplete_checkpoints(self):
        self.service.worker.models.tiers = [
            HardwareTier(
                id="go",
                name="Go",
                vram_gb=8,
                ram_gb=32,
                video_defaults={"duration": 5, "resolution": "480p"},
                models={
                    "llm": "small",
                    "image": None,
                    "video": None,
                    "speech": None,
                    "transcription": None,
                    "formatting": "small",
                },
            )
        ]
        await self.service.download_tier("go", True)
        jobs = list(self.service.worker.jobs.values())
        self.assertEqual([job.request.input.model_id for job in jobs], ["small"])
        self.assertTrue(jobs[0].request.input.license_acknowledged)
        # A queued download is not queued twice.
        await self.service.download_tier("go", True)
        self.assertEqual(len(self.service.worker.jobs), 1)
        self.service.worker.models.models[0].status = "complete"
        self.service._observe(jobs[0], WorkerEvent(state="succeeded"))
        jobs[0].result.set_result({})
        await asyncio.gather(*self.service.tasks.values())
        await self.service.download_tier("go", False)
        self.assertFalse(self.service.worker.jobs)
        with self.assertRaises(InferenceFailure):
            await self.service.download_tier("plus", False)

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

    async def test_lost_download_completion_allows_incomplete_tier_to_resume(self):
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
                self.service.worker.models.tiers = [
                    HardwareTier(
                        id="go",
                        name="Go",
                        vram_gb=8,
                        ram_gb=32,
                        video_defaults={"duration": 5, "resolution": "480p"},
                        models={"llm": "small"},
                    )
                ]
                state = self.service.models_status().models[0]
                self.assertEqual(
                    (state.status, state.error), ("failed", "Worker disconnected.")
                )
                await self.service.download_tier("go", True)
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
            {"feature": "tier_selection", "tier": "go"},
            {"feature": "model_context", "model_id": "small", "context_limit": None},
            {"feature": "whisper_selection", "model": "tiny.en"},
            {"feature": "whisper_formatting", "enabled": False},
        ):
            task = InferenceTask(operation="configure", input=body)
            self.assertEqual(
                InferenceTask.model_validate_json(task.model_dump_json()), task
            )
