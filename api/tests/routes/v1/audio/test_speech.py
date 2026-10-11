from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pydantic import ValidationError
from api.routes.v1.audio.speech import SpeechRequest, list_speech_models


class SpeechTests(unittest.TestCase):
    def test_model_options_use_worker_snapshot(self):
        snapshot = SimpleNamespace(
            models=[
                SimpleNamespace(
                    id="qwen-tts-1.7b-custom",
                    display_name="Worker name",
                    kind="speech",
                    inference_available=True,
                ),
                SimpleNamespace(
                    id="disabled",
                    display_name="Disabled",
                    kind="speech",
                    inference_available=False,
                ),
                SimpleNamespace(
                    id="small",
                    display_name="Small",
                    kind="llm",
                    inference_available=True,
                ),
            ]
        )
        with patch(
            "api.routes.v1.audio.speech.inference.models_status", return_value=snapshot
        ):
            models = list_speech_models()
        self.assertEqual(
            [(model.id, model.name) for model in models],
            [("qwen-tts-1.7b-custom", "Worker name")],
        )
        self.assertEqual(models[0].mode, "custom")
        self.assertIn("Ryan", models[0].speakers)

    def test_voice_options_are_http_input_constraints(self):
        request = SpeechRequest(
            script="Hello", voice={"mode": "custom", "speaker": "Ryan"}
        )
        self.assertEqual(request.model_id, "qwen-tts-1.7b-custom")
        with self.assertRaises(ValidationError):
            SpeechRequest(
                script="Hello", voice={"mode": "custom", "speaker": "Unknown"}
            )
