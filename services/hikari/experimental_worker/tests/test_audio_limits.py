import unittest
from contextlib import nullcontext
from types import SimpleNamespace

from experimental_worker.audio import AudioDurationGuard, generate, validate_context
from experimental_worker.tests import test_audio as audio_tests


class AudioOutputLimitTests(unittest.TestCase):
    def test_all_remaining_audio_context_is_available_without_fixed_reply_reservation(
        self,
    ):
        self.assertEqual(validate_context(1), 32767)
        self.assertEqual(validate_context(100), 32668)
        self.assertEqual(validate_context(32767), 1)
        for count in (0, 32768):
            with self.assertRaises(ValueError):
                validate_context(count)

    def test_long_text_answer_and_transcript_are_not_cut_by_character_limits(self):
        for mode, text in (("answer", "x" * 9000), ("transcribe", "x" * 5000)):
            model, processor, codec = audio_tests.AudioReplyCompletionTests().fixture(7)
            processor.decode.return_value = text
            task = (
                {
                    "mode": mode,
                    "messages": [
                        {"role": "system", "text": "Persona"},
                        {"role": "user", "text": "Hi"},
                    ],
                }
                if mode == "answer"
                else {"mode": mode}
            )
            result = generate(
                task,
                model,
                processor,
                codec,
                SimpleNamespace(no_grad=nullcontext),
                "/tmp/utterance.wav" if mode == "transcribe" else None,
            )
            self.assertEqual(result["text"], text)
            self.assertEqual(model.generate.call_args.kwargs["max_new_tokens"], 32668)

    def test_duration_callback_stops_at_sixty_seconds_without_claiming_completion(self):
        guard = AudioDurationGuard(60 * 24000)
        for _ in range(750):
            guard.put(object())
        with self.assertRaisesRegex(ValueError, "sixty-second"):
            guard.put(object())
        guard.end()

    def test_oversized_audio_codes_fail_before_waveform_decode(self):
        model, processor, codec = audio_tests.AudioReplyCompletionTests().fixture(7)
        model.generate.return_value.audio_lengths = [751]
        processor.decode.return_value = "Hello<|im_end|>"
        with self.assertRaisesRegex(ValueError, "sixty-second"):
            generate(
                audio_tests.AudioReplyCompletionTests().task(),
                model,
                processor,
                codec,
                SimpleNamespace(no_grad=nullcontext),
            )
        codec.decode.assert_not_called()

    def test_remaining_sentence_duration_rejects_before_decode_instead_of_truncation(
        self,
    ):
        model, processor, codec = audio_tests.AudioReplyCompletionTests().fixture(8)
        model.generate.return_value.audio_lengths = [2]
        processor.decode.return_value = "<|audio_out|><|text_end|>"
        with self.assertRaisesRegex(ValueError, "sixty-second"):
            generate(
                {"mode": "tts", "script": "Known words."},
                model,
                processor,
                codec,
                SimpleNamespace(no_grad=nullcontext),
                max_audio_samples=1920,
            )
        codec.decode.assert_not_called()

    def test_generation_callback_aborts_oversized_speech_before_any_waveform_decode(
        self,
    ):
        model, processor, codec = audio_tests.AudioReplyCompletionTests().fixture(7)

        def excessive_audio(**kwargs):
            stream = kwargs["audio_streamer"]
            try:
                for _ in range(751):
                    stream.put(object())
            finally:
                stream.end()

        model.generate.side_effect = excessive_audio
        with self.assertRaisesRegex(ValueError, "sixty-second"):
            generate(
                audio_tests.AudioReplyCompletionTests().task(),
                model,
                processor,
                codec,
                SimpleNamespace(no_grad=nullcontext),
            )
        self.assertEqual(model.generate.call_args.kwargs["max_new_tokens"], 32668)
        processor.decode.assert_not_called()
        codec.decode.assert_not_called()
