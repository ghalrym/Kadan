import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from api.pydantic_models.inference_task import InferenceTask
from api.routes.v1.audio import transcriptions
from api.routes.v1.audio.transcriptions import TranscriptionRequest, get_models


class TranscriptionTests(unittest.TestCase):
    def test_available_whisper_metadata_uses_the_fixed_medium_assignment(self):
        snapshot = SimpleNamespace(
            selected_whisper="tiny.en",
            models=[
                SimpleNamespace(id="whisper-medium", kind="transcription"),
                SimpleNamespace(id="small", kind="llm"),
            ],
        )
        with patch(
            "api.routes.v1.audio.transcriptions.inference.models_status",
            return_value=snapshot,
        ):
            result = get_models()
        self.assertEqual(result.models, ["medium"])
        self.assertEqual(result.selected, "medium")

    def test_fixed_requested_model_is_forwarded_for_worker_resolution(self):
        request = TranscriptionRequest(
            audio="data:audio/wav;base64,AAAA", model="medium"
        )
        self.assertEqual(request.model, "medium")

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
                "model": "medium",
                "formatting_status": "formatted" if enabled else "disabled",
                "formatting_model": "lfm3-1.7b-bf16" if enabled else None,
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
