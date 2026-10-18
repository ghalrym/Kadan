import asyncio
import base64
import io
import unittest
import wave
from unittest.mock import AsyncMock

from api.pydantic_models.chat import ChatMessage
from api.pydantic_models.model_store import ModelStoreStatus
from api.services.inference_errors import InferenceFailure
from api.services.voice_conversation import (
    buffered_pcm,
    capabilities,
    conversation_turn,
    transcribe_window,
)


def checkpoint(model, kind):
    return {
        "id": model,
        "repo_id": model,
        "revision": "test",
        "license": "test",
        "estimated_bytes": 1,
        "kind": kind,
        "display_name": model,
        "license_url": None,
        "license_notice": None,
        "inference_available": True,
        "context_limit": None,
        "architecture_context_limit": None,
        "status": "complete",
        "downloaded_bytes": 1,
        "total_bytes": 1,
        "error": None,
    }


class FakeStream:
    def __init__(self, events):
        self.events = events
        self.closed = False

    async def __aiter__(self):
        for event in self.events:
            yield event

    async def aclose(self):
        self.closed = True


class FakeInference:
    def __init__(self):
        self.snapshot = ModelStoreStatus(
            models=[
                checkpoint("chat", "llm"),
                checkpoint("whisper-medium", "transcription"),
                checkpoint("qwen-tts-1.7b-custom", "speech"),
                checkpoint("drama-3", "speech"),
            ],
            selected_tier="go",
            tiers=[
                {
                    "id": "go",
                    "name": "Go",
                    "vram_gb": 8,
                    "ram_gb": 32,
                    "models": {
                        "llm": "chat",
                        "transcription": "whisper-medium",
                        "speech": "qwen-tts-1.7b-custom",
                    },
                    "video_defaults": {"duration": 5, "resolution": "480p"},
                }
            ],
        )
        self.chat = FakeStream([{"content": "Hello."}, {"finish_reason": "stop"}])
        self.speech = FakeStream(
            [
                {
                    "audio": base64.b64encode(b"\x00\x00" * 441).decode(),
                    "sample_rate": 44100,
                },
                {"finish_reason": "stop", "timing": {"generation_ms": 10}},
            ]
        )
        self.open_chat_stream = AsyncMock(return_value=(self.chat, "chat"))
        self.open_speech_stream = AsyncMock(return_value=self.speech)
        self.submit = AsyncMock(return_value={"raw_text": "hello"})

    def models_status(self):
        return self.snapshot


class VoiceConversationTests(unittest.IsolatedAsyncioTestCase):
    async def test_int4_is_explicitly_unavailable_without_fallback(self):
        service = FakeInference()
        self.assertFalse(capabilities(service)["backends"][2]["available"])
        with self.assertRaisesRegex(InferenceFailure, "INT4"):
            await anext(
                conversation_turn(
                    "go",
                    "drama3-int4",
                    [ChatMessage(role="user", text="hi")],
                    "Ryan",
                    service,
                )
            )
        service.open_chat_stream.assert_not_called()

    async def test_tier_change_is_rejected_before_submission(self):
        service = FakeInference()
        with self.assertRaisesRegex(InferenceFailure, "tier changed"):
            await transcribe_window("audio", "plus", "whisper-medium", service)
        service.submit.assert_not_called()

    async def test_transcription_pins_tier_model_without_formatting_or_download(self):
        service = FakeInference()
        result = await transcribe_window("audio", "go", "whisper-medium", service)
        self.assertEqual(result, {"text": "hello", "model": "whisper-medium"})
        task = service.submit.call_args.args[0]
        self.assertEqual(task.model, "medium")
        self.assertFalse(task.formatting)
        self.assertTrue(task.require_installed)
        self.assertEqual(task.expected_tier, "go")
        with self.assertRaisesRegex(InferenceFailure, "active tier"):
            await transcribe_window("audio", "go", "whisper-tiny", service)

    async def test_uninstalled_models_disable_conversation(self):
        service = FakeInference()
        service.snapshot.models[0].status = "not_downloaded"
        result = capabilities(service)
        self.assertFalse(result["available"])
        self.assertIn("Install chat", result["reason"])
        self.assertFalse(any(item["available"] for item in result["backends"]))

    async def test_native_stream_identity_audio_and_terminal_timing(self):
        service = FakeInference()
        events = [
            event
            async for event in conversation_turn(
                "go", "drama3", [ChatMessage(role="user", text="hi")], "Ryan", service
            )
        ]
        self.assertEqual([item["type"] for item in events], ["text", "audio", "done"])
        self.assertEqual(events[-1]["speech_model"], "drama-3")
        self.assertAlmostEqual(events[-1]["timing"]["audio_duration_seconds"], 0.01)
        self.assertTrue(service.chat.closed and service.speech.closed)
        self.assertEqual(service.open_chat_stream.call_args.args[0].model, "chat")
        self.assertTrue(service.open_speech_stream.call_args.args[0].require_installed)

    async def test_wrong_worker_model_is_closed_without_speech(self):
        service = FakeInference()
        service.open_chat_stream.return_value = service.chat, "other"
        with self.assertRaisesRegex(InferenceFailure, "different conversation model"):
            await anext(
                conversation_turn(
                    "go",
                    "drama3",
                    [ChatMessage(role="user", text="hi")],
                    "Ryan",
                    service,
                )
            )
        self.assertTrue(service.chat.closed)
        service.open_speech_stream.assert_not_called()

    async def test_close_after_audio_cancels_owned_speech_stream(self):
        service = FakeInference()
        turn = conversation_turn(
            "go", "drama3", [ChatMessage(role="user", text="hi")], "Ryan", service
        )
        await anext(turn)
        await anext(turn)
        await turn.aclose()
        self.assertTrue(service.speech.closed)

    async def test_cancellation_during_chat_wait_closes_worker_stream(self):
        service = FakeInference()
        entered = asyncio.Event()

        class WaitingStream(FakeStream):
            async def __aiter__(self):
                entered.set()
                await asyncio.Event().wait()
                yield {}

        stream = WaitingStream([])
        service.open_chat_stream.return_value = stream, "chat"
        turn = conversation_turn(
            "go", "drama3", [ChatMessage(role="user", text="hi")], "Ryan", service
        )
        task = asyncio.create_task(anext(turn))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(stream.closed)
        service.open_speech_stream.assert_not_called()

    async def test_truncated_speech_cannot_report_success(self):
        service = FakeInference()
        service.speech.events.pop()
        events = []
        with self.assertRaisesRegex(InferenceFailure, "terminal event"):
            async for event in conversation_turn(
                "go", "drama3", [ChatMessage(role="user", text="hi")], "Ryan", service
            ):
                events.append(event)
        self.assertNotIn("done", [event["type"] for event in events])
        self.assertTrue(service.speech.closed)

    def test_buffered_qwen_audio_is_framed_without_changing_pcm(self):
        output = io.BytesIO()
        pcm = b"\x01\x00" * 24001
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(pcm)
        chunks = list(
            buffered_pcm({"audio_base64": base64.b64encode(output.getvalue()).decode()})
        )
        self.assertEqual([len(chunk) for chunk, _ in chunks], [48000, 2])
        self.assertEqual(b"".join(chunk for chunk, _ in chunks), pcm)
        with self.assertRaises(InferenceFailure):
            list(buffered_pcm({"audio_base64": "bad"}))
