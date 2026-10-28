import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from api.pydantic_models.chat import ChatMessage
from api.routes.v1.audio.calls import call_event
from api.services.inference_errors import InferenceFailure
from api.services.voice_call import CallModels, VoiceCall, speech_segment


def snapshot():
    return SimpleNamespace(
        defaults=SimpleNamespace(speech="lfm3-audio-2b-bf16"),
        selected_whisper="whisper-medium",
        models=[
            SimpleNamespace(
                id="lfm3-audio-2b-bf16",
                kind="audio",
                status="complete",
                inference_available=True,
            )
        ],
    )


class CallModelTests(unittest.TestCase):
    def test_selects_fixed_audio_assignment_independent_of_whisper_override(self):
        status = snapshot()
        status.selected_whisper = "large-v3"
        self.assertEqual(CallModels.selected(status).audio, "lfm3-audio-2b-bf16")
        self.assertEqual(len(CallModels.selected(status).participants()), 1)

    def test_missing_or_unavailable_checkpoint_requires_user_download(self):
        for unavailable in ("missing", "not_downloaded", "unavailable"):
            with self.subTest(unavailable=unavailable):
                status = snapshot()
                if unavailable == "missing":
                    status.models = []
                elif unavailable == "unavailable":
                    status.models[0].inference_available = False
                else:
                    status.models[0].status = unavailable
                with self.assertRaisesRegex(InferenceFailure, "license"):
                    CallModels.selected(status)

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

    async def test_only_acknowledged_playback_enters_interrupted_history(self):
        call = self.call()
        call.turn = 1
        call.segments = ["Heard.", "Never heard."]
        call.played = 1
        await call.interrupt(2)
        self.assertEqual(call.history, [ChatMessage(role="assistant", text="Heard.")])
        for _ in range(30):
            call.remember("user", "x" * 4000)
        self.assertEqual(len(call.history), 12)
        self.assertTrue(all(len(message.text) == 4000 for message in call.history))

    async def test_unavailable_fixed_model_ends_session_instead_of_switching_models(
        self,
    ):
        status = snapshot()
        call = self.call(SimpleNamespace(models_status=lambda: status))
        status.models[0].status = "not_downloaded"
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
