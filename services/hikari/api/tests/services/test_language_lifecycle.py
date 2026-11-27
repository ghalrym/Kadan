import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from api.pydantic_models.inference_task import InferenceTask
from api.pydantic_models.model_store import CheckpointStatus, ModelStoreStatus
from api.services.inference import InferenceService
from api.services.inference_worker import InferenceJob, JobState


class LanguageLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def service(self):
        service = InferenceService()
        service.worker.models = ModelStoreStatus(
            models=[
                CheckpointStatus(
                    id="inkling-2-mini-iq4-nl",
                    repo_id="ggml-org/Inkling-2-Mini-GGUF",
                    revision="pinned",
                    license="Apache-2.0",
                    estimated_bytes=1,
                    kind="llm",
                    display_name="Inkling 2 Mini",
                    license_url=None,
                    license_notice=None,
                    inference_available=True,
                    status="complete",
                    downloaded_bytes=1,
                    total_bytes=1,
                    error=None,
                    context_limit=65536,
                    architecture_context_limit=1048576,
                )
            ]
        )
        service.worker.workers["text"] = SimpleNamespace(
            kind="llm",
            residency_known=True,
            resident_model=None,
            resident_context={},
        )
        return service

    def test_verified_checkpoint_is_cold_and_has_no_separate_reply_cap(self):
        status = self.service().language_status()
        self.assertEqual(status["state"], "unloaded")
        self.assertEqual(status["model_id"], "inkling-2-mini-iq4-nl")
        self.assertIsNone(status["error"])
        self.assertIsNone(status["max_output_tokens"])
        self.assertIsNone(status["effective_context_limit"])

    def test_resident_native_worker_reports_ready_and_effective_context(self):
        service = self.service()
        worker = service.worker.workers["text"]
        worker.resident_model = "inkling-2-mini-iq4-nl"
        worker.resident_context = {"effective": 65536}
        status = service.language_status()
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["effective_context_limit"], 65536)

    def test_hot_swapped_companion_variant_remains_ready(self):
        service = self.service()
        worker = service.worker.workers["text"]
        worker.resident_model = "inkling-2-mini-abliterated-iq4-nl"
        worker.resident_context = {"effective": 65536}
        status = service.language_status()
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["effective_context_limit"], 65536)

    async def test_generation_is_ready_and_only_the_load_phase_is_loading(self):
        service = self.service()
        job = InferenceJob(InferenceTask(input={"feature": "llm"}), False)
        job.worker = "llm"
        service.worker.jobs[job.request.id] = job
        for state, expected in (
            (JobState.QUEUED, "loading"),
            (JobState.PREPARING, "loading"),
            (JobState.WAITING, "loading"),
            (JobState.LOADING, "loading"),
            (JobState.RUNNING, "ready"),
        ):
            job.state = state
            with self.subTest(state=state):
                self.assertEqual(service.language_status()["state"], expected)

    async def test_generation_stays_ready_with_other_pending_language_work(self):
        service = self.service()
        running = InferenceJob(InferenceTask(input={"feature": "llm"}), False)
        running.worker = "llm"
        running.state = JobState.RUNNING
        pending = InferenceJob(InferenceTask(input={"feature": "llm"}), False)
        pending.worker = "llm"
        service.worker.jobs = {running.request.id: running, pending.request.id: pending}
        for state in (
            JobState.QUEUED,
            JobState.PREPARING,
            JobState.WAITING,
            JobState.LOADING,
        ):
            pending.state = state
            with self.subTest(state=state):
                self.assertEqual(service.language_status()["state"], "ready")
        running.result.set_result({})
        self.assertEqual(service.language_status()["state"], "loading")
        pending.result.set_result({})
        self.assertEqual(service.language_status()["state"], "unloaded")

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
        job.worker = "llm"
        service.worker.jobs[job.request.id] = job
        self.assertEqual(service.language_status()["state"], "loading")
        cleaned = asyncio.Event()

        async def cancel(job_id):
            self.assertEqual(job_id, job.request.id)
            await asyncio.sleep(0)
            job.result.set_result({})
            cleaned.set()

        async def offload(kind):
            self.assertEqual(kind, "llm")
            self.assertTrue(cleaned.is_set())

        with (
            patch.object(service, "cancel", side_effect=cancel),
            patch.object(service.worker, "offload", side_effect=offload),
        ):
            status = await service.unload()
        self.assertEqual(status["state"], "unloaded")
