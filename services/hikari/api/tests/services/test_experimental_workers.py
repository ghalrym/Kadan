import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from api.pydantic_models.inference_task import InferenceTask
from api.pydantic_models.model_store import ModelStoreStatus
from api.services.inference import InferenceService
from api.services.inference_errors import InferenceFailure
from api.services.inference_worker import InferenceJob
from api.services.worker_hub import WorkerHub, target_worker


class FixedRoutingTests(unittest.TestCase):
    def test_default_chat_uses_lfm_worker_without_an_experiment_flag(self):
        with patch.dict("os.environ", {"HIKARI_ENABLE_LFM3_TEXT": "false"}):
            self.assertEqual(
                target_worker(InferenceTask(input={"feature": "llm"})), "lfm3_llm"
            )

    def test_unsupported_language_model_is_rejected(self):
        with self.assertRaises(ValidationError):
            InferenceTask(input={"feature": "llm", "model": "native-moe"})

    def test_manual_load_cannot_claim_residency(self):
        with self.assertRaisesRegex(InferenceFailure, "per request"):
            target_worker(InferenceTask(operation="load", input={"feature": "llm"}))

    def test_empty_catalog_does_not_fabricate_the_default_checkpoint(self):
        hub = WorkerHub()
        hub.primary = "image"
        hub.workers = {
            "image": SimpleNamespace(kind="image", models=ModelStoreStatus(models=[]))
        }
        hub._refresh_models()
        self.assertEqual(hub.models.models, [])
        self.assertEqual(hub.models.selected_model_id, "lfm3-1.7b-bf16")


class ColdLanguageStatusTests(unittest.IsolatedAsyncioTestCase):
    async def test_active_request_is_loading_but_completed_executor_is_not_resident(
        self,
    ):
        service = InferenceService()
        service.worker.models = ModelStoreStatus(models=[])
        job = InferenceJob(InferenceTask(input={"feature": "llm"}), False)
        job.worker = "lfm3_llm"
        service.worker.jobs[job.request.id] = job
        self.assertEqual(service.status()["state"], "loading")
        self.assertIsNone(service.status()["effective_context_limit"])
        job.result.set_result({"text": "complete"})
        self.assertEqual(service.status()["state"], "unloaded")
        self.assertIsNone(service.status()["effective_context_limit"])
        self.assertEqual(service.status()["model_id"], "lfm3-1.7b-bf16")

    async def test_unavailable_checkpoint_cannot_be_reported_as_a_ready_engine(self):
        service = InferenceService()
        service.worker.models = ModelStoreStatus(models=[])
        service.state, service.model_id = "ready", "retired-native-model"
        self.assertEqual(service.status()["state"], "unloaded")
        self.assertIn("unavailable", service.status()["error"])
        self.assertIsNone(service.status()["configured_context_limit"])
