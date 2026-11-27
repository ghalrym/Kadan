import unittest
from types import SimpleNamespace
from unittest.mock import patch

from api.pydantic_models.inference_task import InferenceTask
from api.pydantic_models.model_store import ModelAssignments, ModelStoreStatus
from api.services.worker_hub import WorkerHub, target_worker


class ExperimentalRoutingTests(unittest.TestCase):
    def test_decisions_always_route_to_resident_llm_worker(self):
        task = InferenceTask(
            input={"feature": "decisions", "state": "text", "questions": []}
        )
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(target_worker(task), "llm")
        self.assertEqual(ModelAssignments().decisions, "inkling-2-mini-iq4-nl")
        self.assertNotIn("model_id", task.model_dump()["input"])
        self.assertEqual(
            InferenceTask.model_validate_json(task.model_dump_json()), task
        )

    def test_audio_download_routes_only_to_its_owner_without_a_flag(self):
        hub = WorkerHub()
        audio = SimpleNamespace(
            kind="lfm3_audio",
            ready=SimpleNamespace(is_set=lambda: True),
            computer=SimpleNamespace(id="audio"),
        )
        native = SimpleNamespace(
            kind="speech",
            ready=SimpleNamespace(is_set=lambda: True),
            computer=SimpleNamespace(id="native"),
        )
        hub.workers = {"audio": audio, "native": native}
        request = InferenceTask(
            operation="download",
            input={
                "feature": "models",
                "model_id": "lfm3-audio-2b-bf16",
                "license_acknowledged": True,
            },
        )
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(hub._store_targets(request), [audio])

    def test_chat_and_load_always_route_to_native_llm_worker(self):
        for operation, payload in (
            ("generate", {"feature": "llm", "messages": []}),
            (
                "load",
                {"feature": "llm", "model": "inkling-2-mini-iq4-nl"},
            ),
            (
                "load",
                {"feature": "llm", "model": "inkling-2-mini-q8-0"},
            ),
        ):
            with self.subTest(operation=operation, payload=payload):
                request = InferenceTask(operation=operation, input=payload)
                self.assertEqual(target_worker(request), "llm")

    def test_native_primary_settings_survive_catalog_refresh(self):
        hub = WorkerHub()
        native = ModelStoreStatus(models=[], whisper_formatting=False)
        hub.primary = "native"
        hub.workers = {"native": SimpleNamespace(models=native, kind="llm")}
        hub._refresh_models()
        self.assertFalse(hub.models.whisper_formatting)
        self.assertEqual(hub.models.selected_model_id, "inkling-2-mini-iq4-nl")
        self.assertEqual(hub.models.defaults.llm, "inkling-2-mini-iq4-nl")
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
        self.assertEqual(defaults.llm, "inkling-2-mini-iq4-nl")
        self.assertEqual(defaults.audio_call, "lfm3-audio-2b-bf16")
        self.assertEqual(defaults.transcription, "whisper-medium")
        self.assertEqual(defaults.speech, "qwen-tts-1.7b-custom")
        self.assertEqual(defaults.image, "qwen-image-2.1-turbo")
