import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from api.pydantic_models.inference_task import (
    TranscriptionFormattingTask,
    TranscriptionTask,
)
from api.services import transcription
from api.services.inference_errors import InferenceFailure


def raw_result(status="requested", text="um send it friday no thursday"):
    return {
        "text": text,
        "raw_text": text,
        "language": "en",
        "model": "large-v3",
        "formatting_status": status,
        "formatting_model": None,
    }


class TranscriptionSequenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_returns_raw_without_a_formatter_job(self):
        result = raw_result("disabled")
        body = TranscriptionTask(audio="data:audio/wav;base64,AAAA", formatting=False)
        with patch.object(
            transcription.inference, "submit", new=AsyncMock(return_value=result)
        ) as submit:
            self.assertEqual(await transcription.transcribe(body), result)
        submit.assert_awaited_once_with(body, feature="stt")

    async def test_formatter_receives_raw_transcript_and_preserves_whisper_identity(
        self,
    ):
        raw = raw_result()
        body = TranscriptionTask(audio="data:audio/wav;base64,AAAA")
        for text in ("Send it Thursday.", ""):
            with patch.object(
                transcription.inference,
                "submit",
                new=AsyncMock(
                    side_effect=[raw, {"text": text, "model": "superwhisper/s1-mini"}]
                ),
            ) as submit:
                result = await transcription.transcribe(body)
            self.assertEqual(
                submit.await_args_list[1].args,
                (TranscriptionFormattingTask(text=raw["raw_text"]),),
            )
            self.assertEqual(
                submit.await_args_list[1].kwargs, {"feature": "stt_formatting"}
            )
            self.assertEqual(result["text"], text)
            self.assertEqual(result["raw_text"], raw["raw_text"])
            self.assertEqual(result["model"], "large-v3")
            self.assertEqual(result["formatting_status"], "formatted")
            self.assertEqual(result["formatting_model"], "superwhisper/s1-mini")

    async def test_no_speech_skips_formatter_loading(self):
        with patch.object(
            transcription.inference,
            "submit",
            new=AsyncMock(return_value=raw_result(text="")),
        ) as submit:
            result = await transcription.transcribe(TranscriptionTask(audio="audio"))
        self.assertEqual(submit.await_count, 1)
        self.assertEqual(result["formatting_status"], "empty")
        self.assertEqual(result["text"], "")

    async def test_formatter_failure_returns_raw_with_explicit_warning(self):
        raw = raw_result()
        with patch.object(
            transcription.inference,
            "submit",
            new=AsyncMock(
                side_effect=[raw, InferenceFailure("Insufficient memory", 503)]
            ),
        ):
            result = await transcription.transcribe(TranscriptionTask(audio="audio"))
        self.assertEqual(result["text"], raw["raw_text"])
        self.assertEqual(result["formatting_status"], "failed")
        self.assertEqual(result["formatting_error"], "Insufficient memory")
        self.assertIsNone(result["formatting_model"])

    async def test_oversized_transcript_preserves_raw_with_failure(self):
        raw = raw_result(text="x" * 32001)
        with patch.object(
            transcription.inference, "submit", new=AsyncMock(return_value=raw)
        ) as submit:
            result = await transcription.transcribe(TranscriptionTask(audio="audio"))
        self.assertEqual(submit.await_count, 1)
        self.assertEqual(result["text"], raw["raw_text"])
        self.assertEqual(result["formatting_status"], "failed")

    async def test_cancellation_does_not_turn_into_successful_raw_fallback(self):
        with (
            patch.object(
                transcription.inference,
                "submit",
                new=AsyncMock(side_effect=[raw_result(), asyncio.CancelledError()]),
            ),
            self.assertRaises(asyncio.CancelledError),
        ):
            await transcription.transcribe(TranscriptionTask(audio="audio"))

    async def test_formatter_is_not_queued_before_transcription_finishes(self):
        waiting = asyncio.get_running_loop().create_future()
        calls = []

        async def submit(body, *, feature):
            calls.append(feature)
            if feature == "stt":
                return await waiting
            return {"text": "Thursday.", "model": "superwhisper/s1-mini"}

        with patch.object(transcription.inference, "submit", new=submit):
            task = asyncio.create_task(
                transcription.transcribe(TranscriptionTask(audio="audio"))
            )
            await asyncio.sleep(0)
            self.assertEqual(calls, ["stt"])
            waiting.set_result(raw_result())
            await task
        self.assertEqual(calls, ["stt", "stt_formatting"])
