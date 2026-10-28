import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from api.pydantic_models.inference_task import AudioCallTask
from api.routes.v1.audio.calls import voice_call
from api.services.inference_errors import InferenceFailure
from api.services.voice_call import CallModels, VoiceCall
from api.tests.services.test_voice_call import snapshot


def audio_snapshot():
    return snapshot()


class AudioModelsTests(unittest.TestCase):
    def test_explicit_audio_selection_requires_only_its_approved_download(self):
        status = audio_snapshot()
        models = CallModels.selected(status)
        self.assertEqual(models.participants(), frozenset(("lfm3-audio-2b-bf16",)))
        status.models[0].status = "not_downloaded"
        with self.assertRaisesRegex(InferenceFailure, "license"):
            CallModels.selected(status)

    def test_default_selection_uses_the_fixed_audio_assignment(self):
        self.assertEqual(
            CallModels.selected(audio_snapshot()).audio, "lfm3-audio-2b-bf16"
        )

    def test_audio_task_requires_the_matching_input_and_disallows_tool_history(self):
        for value in (
            {"mode": "transcribe"},
            {"mode": "reply", "audio": "wav"},
            {"mode": "reply", "messages": [{"role": "tool", "text": "result"}]},
        ):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                AudioCallTask(**value)


class AudioCallTests(unittest.IsolatedAsyncioTestCase):
    def call(self, submit):
        return VoiceCall(
            SimpleNamespace(models_status=audio_snapshot, submit=submit),
            CallModels.selected(audio_snapshot()),
            "Saved persona. No tools.",
            AsyncMock(),
        )

    async def test_buffered_audio_uses_session_instruction_and_only_acknowledged_reply_history(
        self,
    ):
        submit = AsyncMock(
            side_effect=[{"text": "Hi"}, {"text": "Hello", "audio_base64": "AAAA"}]
        )
        call = self.call(submit)
        call.turn = 1
        with patch.dict("os.environ", {"HIKARI_LFM3_AUDIO_SINGLE_TURN": "false"}):
            await call.respond("wav", 1)
        tasks = [entry.args[0] for entry in submit.call_args_list]
        self.assertEqual([task.mode for task in tasks], ["transcribe", "reply"])
        # InferenceService excludes unset defaults at the queue boundary. Both
        # stages must still carry the explicitly selected model to the worker.
        self.assertEqual(
            [task.model_dump(exclude_unset=True).get("model_id") for task in tasks],
            ["lfm3-audio-2b-bf16", "lfm3-audio-2b-bf16"],
        )
        self.assertEqual(tasks[1].messages[0].text, "Saved persona. No tools.")
        self.assertEqual([message.role for message in call.history], ["user"])
        events = [entry.args[0] for entry in call.send.call_args_list]
        self.assertEqual(
            [event["type"] for event in events], ["transcript", "text", "audio", "done"]
        )
        await call.interrupt(2)
        self.assertEqual([message.role for message in call.history], ["user"])

    async def test_acknowledged_audio_enters_next_turn(self):
        call = self.call(AsyncMock())
        call.turn = 1
        call.segments = ["Heard reply"]
        call.played = 1
        await call.interrupt(2)
        self.assertEqual(call.history[0].text, "Heard reply")

    async def test_interruption_joins_audio_turn_before_acknowledging(self):
        started, cleaned = asyncio.Event(), asyncio.Event()

        async def submit(*args, **kwargs):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cleaned.set()

        call = self.call(submit)
        call.turn = 1
        with patch.dict("os.environ", {"HIKARI_LFM3_AUDIO_SINGLE_TURN": "false"}):
            call.response = asyncio.create_task(call.respond("wav", 1))
            await started.wait()
            await call.interrupt(2)
        self.assertTrue(cleaned.is_set())
        self.assertEqual(
            [entry.args[0]["type"] for entry in call.send.call_args_list],
            ["interrupted"],
        )

    async def test_missing_default_audio_checkpoint_fails_before_residency(self):
        socket = AsyncMock()
        socket.receive_json.return_value = {"type": "start", "backend": "lfm3"}
        with (
            patch("api.routes.v1.audio.calls.inference") as inference,
        ):
            status = audio_snapshot()
            status.models[0].status = "not_downloaded"
            inference.models_status.return_value = status
            await voice_call(socket)
        inference.models_status.assert_called_once()
        inference.worker.full_duplex.assert_not_called()
        self.assertIn("license", socket.send_json.call_args.args[0]["message"])
