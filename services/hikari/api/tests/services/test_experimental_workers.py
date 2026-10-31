import unittest
from types import SimpleNamespace
from unittest.mock import patch

from api.pydantic_models.inference_task import InferenceTask
from api.pydantic_models.model_store import ModelAssignments, ModelStoreStatus
from api.services.inference import InferenceService
from api.services.inference_errors import InferenceFailure
from api.services.inference_worker import WorkerEvent
from api.services.worker_hub import WorkerHub, target_worker


class ExperimentalRoutingTests(unittest.TestCase):
    def test_decisions_always_use_fixed_d1_without_a_flag(self):
        task = InferenceTask(
            input={"feature": "decisions", "state": "text", "questions": []}
        )
        for env in ({}, {"HIKARI_ENABLE_D1_DECISIONS": "false"}):
            with patch.dict("os.environ", env, clear=True):
                self.assertEqual(target_worker(task), "lfm3_decision")
        self.assertEqual(ModelAssignments().decisions, "d1-2b-cpu-f32")
        self.assertEqual(task.model_dump()["input"]["model_id"], "d1-2b-cpu-f32")
        self.assertEqual(
            InferenceTask.model_validate_json(task.model_dump_json()), task
        )

    def test_d1_download_routes_only_to_its_owner_without_a_flag(self):
        hub = WorkerHub()
        cpu = SimpleNamespace(
            kind="lfm3_decision",
            ready=SimpleNamespace(is_set=lambda: True),
            computer=SimpleNamespace(id="cpu"),
        )
        native = SimpleNamespace(
            kind="speech",
            ready=SimpleNamespace(is_set=lambda: True),
            computer=SimpleNamespace(id="native"),
        )
        hub.workers = {"cpu": cpu, "native": native}
        request = InferenceTask(
            operation="download",
            input={
                "feature": "models",
                "model_id": "d1-2b-cpu-f32",
                "license_acknowledged": True,
            },
        )
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(hub._store_targets(request), [cpu])

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

    def test_default_chat_uses_fixed_lfm_worker(self):
        request = InferenceTask(input={"feature": "llm", "messages": []})
        with patch.dict("os.environ", {"HIKARI_ENABLE_LFM3_TEXT": "true"}):
            self.assertEqual(target_worker(request), "lfm3_llm")

    def test_explicit_model_requires_server_flag(self):
        request = InferenceTask(input={"feature": "llm", "model": "lfm3-1.7b-bf16"})
        with (
            patch.dict("os.environ", {"HIKARI_ENABLE_LFM3_TEXT": "false"}),
            self.assertRaisesRegex(InferenceFailure, "HIKARI_ENABLE_LFM3_TEXT"),
        ):
            target_worker(request)
        with patch.dict("os.environ", {"HIKARI_ENABLE_LFM3_TEXT": "true"}):
            self.assertEqual(target_worker(request), "lfm3_llm")

    def test_advisory_load_cannot_claim_residency(self):
        request = InferenceTask(
            operation="load", input={"feature": "llm", "model": "lfm3-1.7b-bf16"}
        )
        with (
            patch.dict("os.environ", {"HIKARI_ENABLE_LFM3_TEXT": "true"}),
            self.assertRaisesRegex(InferenceFailure, "per request"),
        ):
            target_worker(request)

    def test_native_primary_settings_survive_catalog_refresh(self):
        hub = WorkerHub()
        native = ModelStoreStatus(models=[], whisper_formatting=False)
        hub.primary = "native"
        hub.workers = {"native": SimpleNamespace(models=native, kind="llm")}
        hub._refresh_models()
        self.assertFalse(hub.models.whisper_formatting)
        self.assertEqual(hub.models.selected_model_id, "lfm3-1.7b-bf16")
        self.assertIsNot(hub.models, native)

    def test_default_audio_call_routes_without_opt_in_and_keeps_standalone_speech(self):
        with patch.dict("os.environ", {}, clear=True):
            for request, worker in (
                (
                    {"feature": "audio_call", "mode": "transcribe", "audio": "wav"},
                    "lfm3_audio",
                ),
                ({"feature": "stt", "audio": "wav", "model": "medium"}, "speech"),
                ({"feature": "stt_formatting", "text": "Hello"}, "formatting"),
                (
                    {
                        "feature": "tts",
                        "script": "Hello",
                        "voice": {"mode": "custom", "speaker": "Ryan"},
                    },
                    "tts",
                ),
            ):
                with self.subTest(feature=request["feature"]):
                    self.assertEqual(
                        target_worker(InferenceTask(input=request)), worker
                    )
        defaults = ModelAssignments()
        self.assertEqual(defaults.audio_call, "lfm3-audio-2b-bf16")
        self.assertEqual(defaults.transcription, "whisper-medium")
        self.assertEqual(defaults.speech, "qwen-tts-1.7b-custom")
