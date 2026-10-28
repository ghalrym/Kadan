import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from experimental_worker.audio import conversation, execute
from experimental_worker.speech_voice import GENDERS, LANGUAGES


class StandaloneSpeechTests(unittest.TestCase):
    def test_supported_locales_and_gender_use_the_official_tts_prompt(self):
        self.assertEqual(len(LANGUAGES), 19)
        self.assertEqual(GENDERS, ("female", "male"))
        self.assertEqual(
            conversation({"mode": "tts", "script": "x" * 500})[1]["content"], "x" * 500
        )
        for language in LANGUAGES:
            prompt = conversation(
                {
                    "mode": "tts",
                    "script": "Known words",
                    "language": language,
                    "gender": "male",
                }
            )
            self.assertEqual(
                prompt[0]["content"], f"Perform TTS in {language} using a male voice."
            )
            self.assertEqual(prompt[1]["content"], "Known words")
            self.assertEqual(prompt[2]["content"], "<|audio_start|>")

    def test_unknown_auto_language_and_named_qwen_voices_are_not_advertised_or_accepted(
        self,
    ):
        for changes in (
            {"language": "Auto"},
            {"language": "Russian"},
            {"gender": "Ryan"},
        ):
            with self.assertRaises(ValueError):
                conversation({"mode": "tts", "script": "Known words", **changes})

    def test_public_tts_feature_reuses_one_loaded_model_and_returns_real_clip_metadata(
        self,
    ):
        torch, model, processor, codec = Mock(), Mock(), Mock(), Mock()
        torch.cuda.is_available.return_value = True
        torch.cuda.is_bf16_supported.return_value = True
        model.eval.return_value = model
        processor.audio_tokenizer.to.return_value.eval.return_value = codec
        transformers = SimpleNamespace(
            AutoModelForMultimodalLM=Mock(), AutoProcessor=Mock()
        )
        transformers.AutoModelForMultimodalLM.from_pretrained.return_value = model
        transformers.AutoProcessor.from_pretrained.return_value = processor
        with (
            patch.dict("sys.modules", {"torch": torch, "transformers": transformers}),
            patch(
                "experimental_worker.audio.generate",
                return_value={
                    "text": "",
                    "script": "Known words",
                    "sample_count": 24000,
                    "audio_base64": "AAAA",
                },
            ) as generate,
        ):
            result = execute(
                {
                    "feature": "tts",
                    "model_id": "lfm3-audio-2b-bf16",
                    "script": "Known words",
                    "language": "Japanese",
                    "gender": "male",
                },
                "generate",
                "/tmp/checkpoint",
                Mock(),
            )
        transformers.AutoModelForMultimodalLM.from_pretrained.assert_called_once()
        self.assertEqual(generate.call_args.args[0]["mode"], "tts")
        self.assertEqual(generate.call_args.args[0]["language"], "Japanese")
        self.assertEqual(result["voice"], "Male · Japanese")
        self.assertEqual(result["script"], "Known words")
        self.assertEqual(result["mime_type"], "audio/wav")
        self.assertEqual(result["time"], "1.00s")
        self.assertEqual(result["audio_base64"], "AAAA")
