import unittest
from unittest.mock import AsyncMock

from experimental_worker.call_history import append_call_history

from api.services.voice_call import CallModels, VoiceCall
from api.tests.services.test_experimental_audio_call import audio_snapshot


class AudioModelHistoryTests(unittest.TestCase):
    def test_experimental_history_preserves_complete_messages_until_model_context_check(
        self,
    ):
        call = VoiceCall(
            None, CallModels.selected(audio_snapshot(), "lfm3"), "Persona", AsyncMock()
        )
        text = "x" * 20000
        call.remember("user", text)
        self.assertEqual(call.history[0].text, text)
        self.assertEqual(
            append_call_history([], "user", text, model_bounded=True)[0]["text"], text
        )
        self.assertEqual(append_call_history([], "user", text)[0]["text"], text[:4000])
        for index in range(12):
            call.remember("assistant", str(index))
        self.assertEqual(len(call.history), 12)
        self.assertEqual(call.history[-1].text, "11")
