from types import SimpleNamespace
import unittest
from unittest.mock import patch

from api.routes.v1.audio.transcriptions import get_models, TranscriptionRequest


class TranscriptionTests(unittest.TestCase):
    def test_model_list_and_selection_come_from_worker(self):
        snapshot = SimpleNamespace(
            selected_whisper="tiny.en",
            models=[
                SimpleNamespace(id="whisper-tiny.en", kind="transcription"),
                SimpleNamespace(id="small", kind="llm"),
            ],
        )
        with patch(
            "api.routes.v1.audio.transcriptions.inference.models_status",
            return_value=snapshot,
        ):
            result = get_models()
        self.assertEqual(result.models, ["tiny.en"])
        self.assertEqual(result.selected, "tiny.en")

    def test_requested_model_is_forwarded_for_worker_resolution(self):
        request = TranscriptionRequest(
            audio="data:audio/wav;base64,AAAA", model="turbo"
        )
        self.assertEqual(request.model, "turbo")
