import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from api.pydantic_models.inference_task import InferenceTask
from api.pydantic_models.model_store import ModelStoreStatus
from api.services.inference import InferenceService
from api.services.inference_worker import InferenceJob
from api.pydantic_models.model_store import CheckpointStatus


class LanguageLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def service(self):
        service = InferenceService()
        service.worker.models = ModelStoreStatus(
            models=[
                CheckpointStatus(
                    id="lfm3-1.7b-bf16",
                    repo_id="test/text",
                    revision="pinned",
                    license="test",
                    estimated_bytes=1,
                    kind="llm",
                    display_name="LFM3",
                    license_url=None,
                    license_notice=None,
                    inference_available=True,
                    status="complete",
                    downloaded_bytes=1,
                    total_bytes=1,
                    error=None,
                    context_limit=262144,
                    architecture_context_limit=262144,
                )
            ]
        )
        service.worker.workers["text"] = SimpleNamespace(
            kind="lfm3_llm", residency_known=True
        )
        return service

    def test_verified_checkpoint_is_cold_and_has_no_separate_reply_cap(self):
        status = self.service().language_status()
        self.assertEqual(status["state"], "unloaded")
        self.assertIsNone(status["error"])
        self.assertIsNone(status["max_output_tokens"])
        self.assertIsNone(status["effective_context_limit"])

    def test_missing_checkpoint_or_worker_and_unconfirmed_cleanup_are_errors(self):
        for missing in ("worker", "checkpoint", "cleanup"):
            service = self.service()
            if missing == "worker":
                service.worker.workers.clear()
            elif missing == "checkpoint":
                service.worker.models.models[0].status = "not_downloaded"
            else:
                service.worker.workers["text"].residency_known = False
            with self.subTest(missing=missing):
                status = service.language_status()
                self.assertEqual(status["state"], "error")
                self.assertTrue(status["error"])

    async def test_unload_awaits_cancellation_before_requesting_release(self):
        service = self.service()
        job = InferenceJob(InferenceTask(input={"feature": "llm"}), False)
        job.worker = "lfm3_llm"
        service.worker.jobs[job.request.id] = job
        self.assertEqual(service.language_status()["state"], "loading")
        cleaned = asyncio.Event()

        async def cancel(job_id):
            self.assertEqual(job_id, job.request.id)
            await asyncio.sleep(0)
            job.result.set_result({})
            cleaned.set()

        async def offload(kind):
            self.assertEqual(kind, "lfm3_llm")
            self.assertTrue(cleaned.is_set())

        with (
            patch.object(service, "cancel", side_effect=cancel),
            patch.object(service.worker, "offload", side_effect=offload),
        ):
            status = await service.unload()
        self.assertEqual(status["state"], "unloaded")
