import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from api.routes.v1.audio.calls import CallStart
from api.services.inference_errors import InferenceFailure
from api.services.voice_call import CallModels, VoiceCall


def status(complete=True):
    return SimpleNamespace(
        defaults=SimpleNamespace(speech="lfm3-audio-2b-bf16"),
        models=[
            SimpleNamespace(
                id="lfm3-audio-2b-bf16",
                status="complete" if complete else "not_downloaded",
                inference_available=True,
            )
        ],
    )


class FixedAudioCallTests(unittest.IsolatedAsyncioTestCase):
    def test_call_model_is_the_fixed_audio_assignment_without_tier_or_qwen_fallback(
        self,
    ):
        selected = CallModels.selected(status())
        self.assertEqual(selected.model_dump(), {"audio": "lfm3-audio-2b-bf16"})
        self.assertEqual(selected.participants(), frozenset(("lfm3-audio-2b-bf16",)))
        with self.assertRaises(InferenceFailure):
            CallModels.selected(status(False))
        invalid = status()
        invalid.defaults.speech = "qwen-tts-1.7b-custom"
        with self.assertRaises(InferenceFailure):
            CallModels.selected(invalid)

    def test_call_start_defaults_to_lfm_and_rejects_the_retired_native_choice(self):
        self.assertEqual(CallStart(type="start").backend, "lfm3")
        with self.assertRaises(ValidationError):
            CallStart(type="start", backend="native")

    async def test_default_response_uses_one_audio_turn_and_no_qwen_native_consumer(
        self,
    ):
        service = SimpleNamespace(models_status=status)
        call = VoiceCall(
            service, CallModels.selected(status()), "Saved persona", AsyncMock()
        )
        with (
            patch.object(call, "respond_audio_turn", new=AsyncMock()) as respond,
            patch.dict("os.environ", {}, clear=True),
        ):
            await call.respond("wav", 1)
        respond.assert_awaited_once_with("wav", 1)
