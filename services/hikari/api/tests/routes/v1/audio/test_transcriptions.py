import unittest
from unittest.mock import AsyncMock, patch

from api.pydantic_models.inference_task import InferenceTask
from api.routes.v1.audio import transcriptions
from api.routes.v1.audio.transcriptions import TranscriptionRequest
from pydantic import ValidationError


class TranscriptionTests(unittest.TestCase):
    def test_only_fixed_whisper_checkpoint_is_accepted(self):
        self.assertEqual(
            TranscriptionRequest(
                audio="data:audio/wav;base64,AAAA", model="medium"
            ).model,
            "medium",
        )
        with self.assertRaises(ValidationError):
            TranscriptionRequest(audio="data:audio/wav;base64,AAAA", model="turbo")

    def test_formatting_preference_is_deferred_to_worker_unless_overridden(self):
        for fields in (
            {},
            {"formatting": None},
            {"formatting": False},
            {"formatting": True},
        ):
            body = TranscriptionRequest(audio="data:audio/wav;base64,AAAA", **fields)
            task = InferenceTask(
                input={"feature": "stt", **body.model_dump(exclude_unset=True)}
            )
            serialized = task.model_dump()["input"]
            if not fields:
                self.assertNotIn("formatting", serialized)
            else:
                self.assertEqual(serialized["formatting"], fields["formatting"])


class TranscriptionResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_formatted_and_raw_text_are_preserved(self):
        async def infer(_request, submission):
            return await submission

        for enabled in (False, True):
            raw = "  Pay  $1.25, not $125.  "
            result = {
                "text": raw.strip().replace("  ", " ") if enabled else raw,
                "raw_text": raw,
                "language": "en",
                "model": "large-v3",
                "formatting_status": "formatted" if enabled else "disabled",
                "formatting_model": "superwhisper/s1-mini" if enabled else None,
                "formatting_error": None,
            }
            body = TranscriptionRequest(
                audio="data:audio/wav;base64,AAAA", formatting=enabled
            )
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
