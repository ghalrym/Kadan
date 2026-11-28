import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from api.pydantic_models.inference_task import InferenceTask
from api.routes.v1.audio import transcriptions
from api.routes.v1.audio.transcriptions import TranscriptionRequest


class TranscriptionTests(unittest.TestCase):
    def test_only_fixed_qwen3_asr_checkpoint_is_accepted(self):
        self.assertEqual(
            TranscriptionRequest(
                audio="data:audio/wav;base64,AAAA", model="qwen3-asr-1.7b"
            ).model,
            "qwen3-asr-1.7b",
        )
        with self.assertRaises(ValidationError):
            TranscriptionRequest(audio="data:audio/wav;base64,AAAA", model="medium")

    def test_task_serializes_audio_and_optional_language(self):
        body = TranscriptionRequest(audio="data:audio/wav;base64,AAAA", language="en")
        task = InferenceTask(
            input={"feature": "stt", **body.model_dump(exclude_unset=True)}
        )
        serialized = task.model_dump()["input"]
        self.assertEqual(serialized["audio"], "data:audio/wav;base64,AAAA")
        self.assertEqual(serialized["language"], "en")


class TranscriptionResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_transcript_is_returned(self):
        async def infer(_request, submission):
            return await submission

        result = {
            "text": "Pay $1.25, not $125.",
            "language": "en",
            "model": "qwen3-asr-1.7b",
        }
        body = TranscriptionRequest(audio="data:audio/wav;base64,AAAA")
        with (
            patch.object(
                transcriptions,
                "transcribe",
                new=AsyncMock(return_value=result),
            ) as submit,
            patch.object(transcriptions, "infer", new=infer),
        ):
            response = await transcriptions.transcribe_audio(body, object())
        self.assertEqual(response.model_dump(), result)
        submit.assert_awaited_once_with(body)
