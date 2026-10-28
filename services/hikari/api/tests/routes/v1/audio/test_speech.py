import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from api.pydantic_models.inference_task import SpeechTask
from api.services.inference import InferenceService
from api.services.inference_worker import InferenceJob
from api.services.worker_hub import target_worker
from api.routes.v1.audio import speech
from api.routes.v1.audio.speech import SpeechRequest, list_speech_models


class SpeechTests(unittest.IsolatedAsyncioTestCase):
    def test_model_options_expose_only_real_lfm_languages_and_gender_choices(self):
        snapshot = SimpleNamespace(
            models=[
                SimpleNamespace(
                    id="lfm3-audio-2b-bf16",
                    display_name="LFM3 Audio",
                    kind="audio",
                    inference_available=True,
                ),
                SimpleNamespace(
                    id="qwen-tts-1.7b-custom",
                    display_name="Retired",
                    kind="speech",
                    inference_available=True,
                ),
            ]
        )
        with patch.object(speech.inference, "models_status", return_value=snapshot):
            models = list_speech_models()
        self.assertEqual([model.id for model in models], ["lfm3-audio-2b-bf16"])
        self.assertEqual(len(models[0].languages), 19)
        self.assertEqual(models[0].genders, ["female", "male"])
        self.assertEqual(models[0].default_language, "American English")
        self.assertFalse(hasattr(models[0], "speakers"))

    def test_language_gender_and_retired_named_voice_are_http_input_constraints(self):
        request = SpeechRequest(script="Hello", language="Japanese", gender="male")
        self.assertEqual(request.model_id, "lfm3-audio-2b-bf16")
        for changes in (
            {"language": "Auto"},
            {"language": "Russian"},
            {"gender": "Ryan"},
            {"voice": {"mode": "custom", "speaker": "Ryan"}},
            {"model_id": "qwen-tts-1.7b-custom"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                SpeechRequest(script="Hello", **changes)
        self.assertEqual(len(SpeechRequest(script="x" * 40000).script), 40000)

    async def test_speech_submission_carries_explicit_checkpoint_and_selected_locale(
        self,
    ):
        body = SpeechRequest(
            script="Known words", language="British English", gender="female"
        )
        result = {
            "voice": "Female · British English",
            "meta": "LFM3 Audio BF16",
            "script": "Known words",
            "time": "1.00s",
            "mime_type": "audio/wav",
            "audio_base64": "AAAA",
        }

        async def await_queue(request, operation):
            return await operation

        with (
            patch.object(
                speech.inference, "submit", new=AsyncMock(return_value=result)
            ) as submit,
            patch.object(speech, "infer", await_queue),
        ):
            response = await speech.generate_speech(body, object())
        queued = submit.call_args.args[0].model_dump(exclude_unset=True)
        self.assertEqual(queued["model_id"], "lfm3-audio-2b-bf16")
        self.assertEqual(
            (queued["language"], queued["gender"]), ("British English", "female")
        )
        self.assertEqual(submit.call_args.kwargs, {"feature": "tts"})
        self.assertEqual(response.audio.audio_base64, "AAAA")

    async def test_public_defaults_reach_typed_audio_worker_without_tiers_or_flags(
        self,
    ):
        service = InferenceService()
        queued = []
        result = {
            "voice": "Female · American English",
            "meta": "LFM3 Audio BF16 · WAV 24 kHz",
            "script": "Known words",
            "time": "1.00s",
            "mime_type": "audio/wav",
            "audio_base64": "AAAA",
        }

        async def accept(request, *, streaming):
            queued.append(request)
            self.assertFalse(streaming)
            self.assertIsInstance(request.input, SpeechTask)
            self.assertEqual(target_worker(request), "lfm3_audio")
            self.assertEqual(request.input.model_id, "lfm3-audio-2b-bf16")
            self.assertEqual(request.input.language, "American English")
            self.assertEqual(request.input.gender, "female")
            self.assertNotIn("voice", request.input.model_dump())
            job = InferenceJob(request, streaming=False)
            service.worker.jobs[request.id] = job
            job.result.set_result(result)
            return job

        async def await_queue(request, operation):
            return await operation

        with (
            patch.object(speech, "inference", service),
            patch.object(service.worker, "submit", side_effect=accept),
            patch.object(speech, "infer", await_queue),
            patch.dict("os.environ", {}, clear=True),
        ):
            response = await speech.generate_speech(
                SpeechRequest(script="Known words"), object()
            )
        self.assertEqual(len(queued), 1)
        self.assertEqual(
            queued[0].input.model_dump(exclude_unset=True)["model_id"],
            "lfm3-audio-2b-bf16",
        )
        self.assertEqual(response.audio.model_dump(), result)
        self.assertEqual(service.worker.jobs, {})
