import unittest
from unittest.mock import AsyncMock, patch

from api.pydantic_models.inference_task import TranscriptionTask
from api.services import transcription


class TranscriptionSequenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_transcription_submits_directly_to_stt_worker(self):
        result = {
            "text": "Send it Thursday.",
            "language": "en",
            "model": "qwen3-asr-1.7b",
        }
        body = TranscriptionTask(audio="data:audio/wav;base64,AAAA")
        with patch.object(
            transcription.inference, "submit", new=AsyncMock(return_value=result)
        ) as submit:
            self.assertEqual(await transcription.transcribe(body), result)
        submit.assert_awaited_once_with(body, feature="stt")
