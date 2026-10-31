import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from api.pydantic_models.chat import ChatMessage
from api.pydantic_models.model_store import ModelStoreStatus
from api.routes.v1.audio.calls import call_event
from api.services.inference_errors import InferenceFailure
from api.services.voice_call import CallModels, VoiceCall


def snapshot():
    kinds = {"audio": "lfm3-audio-2b-bf16"}
    return ModelStoreStatus.model_validate(
        {
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


class CallModelTests(unittest.TestCase):
    def test_uses_only_fixed_audio_call_participant(self):
        models = CallModels.selected(snapshot())
        self.assertEqual(models.audio, "lfm3-audio-2b-bf16")

    def test_unavailable_checkpoint_is_actionable_without_downloading(self):
        for state, available in (("not_downloaded", True), ("complete", False)):
            with self.subTest(state=state, available=available):
                status = snapshot()
                status.models[0].status = state
                status.models[0].inference_available = available
                message = "license.*download" if available else "worker is unavailable"
                with self.assertRaisesRegex(InferenceFailure, message):
                    CallModels.selected(status)

    def test_missing_audio_worker_is_actionable(self):
        with self.assertRaisesRegex(InferenceFailure, "worker is unavailable"):
            CallModels.selected(ModelStoreStatus(models=[]))


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
        self.assertTrue(all(message.text == "x" * 4000 for message in call.history))

    async def test_lost_checkpoint_ends_session(self):
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

    async def test_end_joins_current_response_before_returning(self):
        call = self.call()
        cleaned = asyncio.Event()

        async def response():
            try:
                await asyncio.Future()
            finally:
                cleaned.set()

        call.response = asyncio.create_task(response())
        await asyncio.sleep(0)
        receive = AsyncMock(return_value=call_event.validate_python({"type": "end"}))
        await call.run(receive)
        self.assertTrue(cleaned.is_set())
        self.assertIsNone(call.response)

    async def test_out_of_order_playback_acknowledgement_is_rejected(self):
        call = self.call()
        call.turn = 1
        call.segments = ["First", "Second"]
        receive = AsyncMock(
            return_value=call_event.validate_python(
                {"type": "played", "turn": 1, "segment": 1}
            )
        )
        with self.assertRaisesRegex(InferenceFailure, "playback acknowledgement"):
            await call.run(receive)
        self.assertEqual(call.played, 0)
