import unittest
from types import SimpleNamespace
from unittest.mock import patch

from api.pydantic_models.inference_task import InferenceTask
from api.pydantic_models.model_store import ModelStoreStatus
from api.services.inference import InferenceService
from api.services.inference_errors import InferenceFailure
from api.services.inference_worker import WorkerEvent
from api.services.worker_hub import WorkerHub, target_worker


class ExperimentalRoutingTests(unittest.TestCase):
    def test_d1_is_opt_in_and_never_replaces_default_laya_or_chat(self):
        body = {
            "feature": "decisions",
            "state": "State",
            "questions": [{"key": "q", "type": "Noul", "instructions": "True?"}],
        }
        self.assertEqual(target_worker(InferenceTask(input=body)), "decision")
        request = InferenceTask(input={**body, "model": "d1-2b-bf16"})
        with patch.dict("os.environ", {"HIKARI_ENABLE_D1": "false"}):
            with self.assertRaisesRegex(InferenceFailure, "HIKARI_ENABLE_D1"):
                target_worker(request)
        with patch.dict("os.environ", {"HIKARI_ENABLE_D1": "true"}):
            self.assertEqual(target_worker(request), "d1_decision")
            self.assertEqual(target_worker(InferenceTask(input=body)), "decision")
            with self.assertRaisesRegex(InferenceFailure, "structured decisions"):
                target_worker(
                    InferenceTask(input={"feature": "llm", "model": "d1-2b-bf16"})
                )

    def test_d1_catalog_does_not_replace_native_tiers_and_is_hidden_when_disabled(self):
        from experimental_worker.d1 import MODEL

        hub = WorkerHub()
        native = ModelStoreStatus(
            models=[], selected_tier="plus", selected_model_id="native-moe"
        )
        observed = {
            **MODEL,
            "status": "not_downloaded",
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "error": None,
            "inference_available": True,
        }
        hub.primary = "native"
        hub.workers = {
            "native": SimpleNamespace(models=native, kind="llm"),
            "d1": SimpleNamespace(
                models=ModelStoreStatus(models=[observed]), kind="d1_decision"
            ),
        }
        for enabled in ("false", "true"):
            with (
                self.subTest(enabled=enabled),
                patch.dict("os.environ", {"HIKARI_ENABLE_D1": enabled}),
            ):
                hub._refresh_models()
                self.assertEqual(hub.models.selected_tier, "plus")
                self.assertEqual(hub.models.selected_model_id, "native-moe")
                self.assertEqual(
                    [model.id for model in hub.models.models],
                    [] if enabled == "false" else [MODEL["id"]],
                )

    def test_experimental_events_do_not_replace_native_language_status(self):
        service = InferenceService()
        service.state, service.model_id, service.context = (
            "ready",
            "native-moe",
            {"effective": 2048},
        )
        request = InferenceTask(input={"feature": "llm", "model": "lfm3-1.7b-bf16"})
        job = SimpleNamespace(
            request=request, worker="lfm3_llm", image_progress=None, progress=None
        )
        for state in ("loading", "succeeded", "failed", "cancelled"):
            with self.subTest(state=state):
                service._observe(
                    job,
                    WorkerEvent(
                        state=state, model="lfm3-1.7b-bf16", context={"effective": 8192}
                    ),
                )
                self.assertEqual(service.state, "ready")
                self.assertEqual(service.model_id, "native-moe")
                self.assertEqual(service.context, {"effective": 2048})
                self.assertIsNone(service.error)

    def test_default_chat_keeps_native_worker(self):
        request = InferenceTask(input={"feature": "llm", "messages": []})
        self.assertEqual(target_worker(request), "llm")

    def test_explicit_model_requires_server_flag(self):
        request = InferenceTask(input={"feature": "llm", "model": "lfm3-1.7b-bf16"})
        with patch.dict("os.environ", {"HIKARI_ENABLE_LFM3_TEXT": "false"}):
            with self.assertRaisesRegex(InferenceFailure, "HIKARI_ENABLE_LFM3_TEXT"):
                target_worker(request)
        with patch.dict("os.environ", {"HIKARI_ENABLE_LFM3_TEXT": "true"}):
            self.assertEqual(target_worker(request), "lfm3_llm")

    def test_advisory_load_cannot_claim_residency(self):
        request = InferenceTask(
            operation="load", input={"feature": "llm", "model": "lfm3-1.7b-bf16"}
        )
        with patch.dict("os.environ", {"HIKARI_ENABLE_LFM3_TEXT": "true"}):
            with self.assertRaisesRegex(InferenceFailure, "per request"):
                target_worker(request)

    def test_native_primary_settings_survive_catalog_refresh(self):
        hub = WorkerHub()
        native = ModelStoreStatus(
            models=[], selected_tier="plus", selected_model_id="native-moe"
        )
        hub.primary = "native"
        hub.workers = {"native": SimpleNamespace(models=native, kind="llm")}
        hub._refresh_models()
        self.assertEqual(hub.models.selected_tier, "plus")
        self.assertEqual(hub.models.selected_model_id, "native-moe")
        self.assertIsNot(hub.models, native)
