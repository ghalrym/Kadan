import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from api.pydantic_models.chat import ChatMessage
from api.pydantic_models.model_store import ModelStoreStatus
from api.routes.v1.audio.calls import call_event
from api.services.inference_errors import InferenceFailure
from api.services.voice_call import CallModels, VoiceCall, speech_segment


def snapshot():
    kinds = {
        "llm": "chat",
        "transcription": "whisper-medium",
        "speech": "qwen-tts-1.7b-custom",
    }
    return ModelStoreStatus.model_validate(
        {
            "selected_tier": "go",
            "tiers": [
                {
                    "id": "go",
                    "name": "Go",
                    "vram_gb": 8,
                    "ram_gb": 16,
                    "video_defaults": {"duration": 5, "resolution": "480p"},
                    "models": kinds,
                }
            ],
            "models": [
                {
                    "id": model,
                    "kind": kind,
                    "repo_id": "local",
                    "revision": "revision",
                    "license": "license",
                    "estimated_bytes": 1,
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
                for kind, model in kinds.items()
            ],
        }
    )


class Stream:
    def __init__(self, events):
        self.events = events
        self.closed = False

    async def __aiter__(self):
        for event in self.events:
            yield event

    async def aclose(self):
        self.closed = True


class CallModelTests(unittest.TestCase):
    def test_selects_all_three_tier_assignments_not_saved_whisper_override(self):
        status = snapshot()
        status.selected_whisper = "large-v3"
        self.assertEqual(CallModels.selected(status).transcription, "whisper-medium")
        self.assertEqual(len(CallModels.selected(status).participants()), 3)

    def test_missing_tier_or_checkpoint_is_actionable_without_downloading(self):
        status = snapshot()
        for kind in (None, "llm", "transcription", "speech"):
            with self.subTest(kind=kind):
                value = status.model_copy(deep=True)
                if kind is None:
                    value.selected_tier = None
                else:
                    next(
                        model for model in value.models if model.kind == kind
                    ).status = "not_downloaded"
                with self.assertRaises(InferenceFailure):
                    CallModels.selected(value)

    def test_synthesis_segments_are_bounded_and_keep_remaining_text(self):
        self.assertEqual(speech_segment("Hello. More"), ("Hello.", "More"))
        self.assertEqual(speech_segment("unfinished"), ("", "unfinished"))
        self.assertEqual(speech_segment("unfinished", final=True), ("unfinished", ""))
        segment, rest = speech_segment("x" * 300)
        self.assertEqual((len(segment), len(rest)), (240, 60))


class VoiceCallTests(unittest.IsolatedAsyncioTestCase):
    def call(self, service=None):
        status = snapshot()
        service = service or SimpleNamespace(models_status=lambda: status)
        return VoiceCall(
            service,
            CallModels.selected(status),
            "Short spoken replies. No tools.",
            AsyncMock(),
        )

    async def test_streams_text_and_synthesized_sentences_through_existing_service(
        self,
    ):
        stream = Stream(
            [
                {"content": "Hello. "},
                {"content": "How are you?"},
                {"finish_reason": "stop"},
            ]
        )
        submitted = []

        async def submit(body, *, feature):
            submitted.append(body)
            return {"text": "Hi"} if feature == "stt" else {"audio_base64": "AAAA"}

        service = SimpleNamespace(
            models_status=snapshot,
            submit=submit,
            open_chat_stream=AsyncMock(return_value=(stream, "chat")),
        )
        call = self.call(service)
        call.turn = 1
        await call.respond("data:audio/wav;base64,AAAA", 1)
        self.assertEqual(submitted[0].model, "medium")
        self.assertFalse(submitted[0].formatting)
        self.assertEqual(
            [body.script for body in submitted[1:]], ["Hello.", "How are you?"]
        )
        self.assertTrue(
            all(body.model_id == "qwen-tts-1.7b-custom" for body in submitted[1:])
        )
        chat = service.open_chat_stream.call_args.args[0]
        self.assertEqual(chat.model, "chat")
        self.assertNotIn("tools", chat.model_dump())
        events = [entry.args[0] for entry in call.send.call_args_list]
        self.assertEqual(
            [event["segment"] for event in events if event["type"] == "audio"], [0, 1]
        )
        self.assertEqual(events[-1]["type"], "done")
        self.assertTrue(stream.closed)

    async def test_interrupt_joins_old_inference_and_does_not_send_stale_audio(self):
        started, cleaned = asyncio.Event(), asyncio.Event()

        async def submit(*args, **kwargs):
            started.set()
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                cleaned.set()

        call = self.call(SimpleNamespace(models_status=snapshot, submit=submit))
        call.turn = 1
        call.response = asyncio.create_task(call.respond("wav", 1))
        await started.wait()
        await call.interrupt(2)
        self.assertTrue(cleaned.is_set())
        await call.emit(1, type="audio", audio="stale")
        self.assertEqual(
            [entry.args[0]["type"] for entry in call.send.call_args_list],
            ["interrupted"],
        )

    async def test_interrupt_cancels_synthesis_and_closes_chat_stream(self):
        speaking, cleaned = asyncio.Event(), asyncio.Event()
        stream = Stream([{"content": "Hello."}, {"finish_reason": "stop"}])

        async def submit(body, *, feature):
            if feature == "stt":
                return {"text": "Hi"}
            speaking.set()
            try:
                await asyncio.Future()
            finally:
                cleaned.set()

        call = self.call(
            SimpleNamespace(
                models_status=snapshot,
                submit=submit,
                open_chat_stream=AsyncMock(return_value=(stream, "chat")),
            )
        )
        call.turn = 1
        call.response = asyncio.create_task(call.respond("wav", 1))
        await speaking.wait()
        await call.interrupt(2)
        self.assertTrue(cleaned.is_set())
        self.assertTrue(stream.closed)
        self.assertFalse(
            any(entry.args[0]["type"] == "audio" for entry in call.send.call_args_list)
        )

    async def test_only_acknowledged_playback_enters_interrupted_history(self):
        call = self.call()
        call.turn = 1
        call.segments = ["Heard.", "Never heard."]
        call.played = 1
        await call.interrupt(2)
        self.assertEqual(call.history, [ChatMessage(role="assistant", text="Heard.")])
        for _ in range(30):
            call.remember("user", "x" * 4000)
        self.assertLessEqual(sum(len(message.text) for message in call.history), 12000)

    async def test_tier_change_ends_session_instead_of_switching_models(self):
        status = snapshot()
        call = self.call(SimpleNamespace(models_status=lambda: status))
        status.selected_tier = None
        with self.assertRaises(InferenceFailure):
            call.check_models()

    async def test_disconnect_joins_current_response(self):
        call = self.call()
        cleaned = asyncio.Event()

        async def response():
            try:
                await asyncio.Future()
            finally:
                cleaned.set()

        call.response = asyncio.create_task(response())
        await asyncio.sleep(0)
        receive = AsyncMock(side_effect=RuntimeError("disconnected"))
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            await call.run(receive)
        self.assertTrue(cleaned.is_set())

    async def test_duplicate_utterance_is_rejected_and_owned_response_is_cancelled(
        self,
    ):
        call = self.call()
        call.turn = 1
        call.accepted = True
        receive = AsyncMock(
            return_value=call_event.validate_python(
                {"type": "utterance", "turn": 1, "audio": "data:audio/wav;base64,AAAA"}
            )
        )
        with self.assertRaisesRegex(InferenceFailure, "Interrupt"):
            await call.run(receive)

    async def test_late_acknowledgement_does_not_block_subsequent_end(self):
        call = self.call()
        call.turn = 2
        incoming = iter(
            (
                call_event.validate_python({"type": "played", "turn": 1, "segment": 0}),
                call_event.validate_python({"type": "end"}),
            )
        )
        await asyncio.wait_for(
            call.run(AsyncMock(side_effect=lambda: next(incoming))), 1
        )
        self.assertEqual(call.played, 0)

    async def test_synthesis_admission_failure_keeps_actionable_error_and_closes_stream(
        self,
    ):
        stream = Stream([{"content": "Hi."}, {"finish_reason": "stop"}])

        async def submit(body, *, feature):
            if feature == "stt":
                return {"text": "Hello"}
            raise InferenceFailure("Needs 4 GiB VRAM", 507)

        call = self.call(
            SimpleNamespace(
                models_status=snapshot,
                submit=submit,
                open_chat_stream=AsyncMock(return_value=(stream, "chat")),
            )
        )
        with self.assertRaisesRegex(InferenceFailure, "Needs 4 GiB"):
            await call.respond("wav", 0)
        self.assertTrue(stream.closed)
