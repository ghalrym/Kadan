import base64
import io
import unittest
import wave
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from experimental_worker.audio import (
    MAX_CONTEXT_TOKENS,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    audio_bytes,
    conversation,
    generate,
    public_text,
    validate_context,
)


def wav_uri(frames=1600, rate=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(2)
        sink.setframerate(rate)
        sink.writeframes(b"\x00\x00" * frames)
    return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode()


class Inputs(dict):
    def __init__(self, count):
        super().__init__(input_ids=SimpleNamespace(shape=(1, count)))

    def to(self, device):
        return self


class Sequences:
    def __init__(self, tokens=None):
        self.tokens = [7] if tokens is None else tokens

    def __getitem__(self, item):
        return self.tokens


class AudioAdapterTests(unittest.TestCase):
    def test_pcm_duration_and_format_are_checked_without_loading_models(self):
        self.assertTrue(audio_bytes(wav_uri()))
        for value in (
            wav_uri(frames=320001),
            wav_uri(rate=24000),
            "https://audio",
            "data:audio/wav;base64,invalid",
        ):
            with self.subTest(value=value[:40]), self.assertRaises(ValueError):
                audio_bytes(value)

    def test_reply_preserves_persona_and_history_and_prefills_mixed_output(self):
        messages = [
            {"role": "system", "text": "Persona"},
            {"role": "assistant", "text": "Heard"},
            {"role": "user", "text": "Hello"},
        ]
        result = conversation({"mode": "reply", "messages": messages})
        self.assertEqual(result[1], {"role": "assistant", "content": "Heard"})
        self.assertTrue(result[0]["content"].endswith("Persona"))
        self.assertEqual(result[-1]["content"], "<|mixed_start|>")
        self.assertEqual(messages[0]["text"], "Persona")

    def test_overlong_tokenized_prompt_fails_instead_of_truncating_or_generating(self):
        processor = Mock()
        processor.apply_chat_template.return_value = Inputs(MAX_INPUT_TOKENS + 1)
        model = Mock()
        with self.assertRaisesRegex(ValueError, "32768-token"):
            generate(
                {"mode": "transcribe"}, model, processor, Mock(), Mock(), "/tmp/wav"
            )
        model.generate.assert_not_called()

    def test_asr_suppresses_speech_and_returns_only_decoded_text(self):
        processor = Mock()
        processor.apply_chat_template.return_value = Inputs(100)
        processor.decode.return_value = "Hello"
        model = Mock()
        model.config.text_config.eos_token_id = 7
        model.config.text_end_token_id = 8
        model.generate.return_value = SimpleNamespace(sequences=Sequences())
        result = generate(
            {"mode": "transcribe"},
            model,
            processor,
            Mock(),
            SimpleNamespace(no_grad=nullcontext),
            "/tmp/wav",
        )
        self.assertEqual(result, {"text": "Hello"})
        self.assertTrue(model.generate.call_args.kwargs["suppress_audio_out"])
        self.assertTrue(model.generate.call_args.kwargs["suppress_mixed"])

    def test_incomplete_text_is_not_published_as_completed_speech(self):
        processor = Mock()
        processor.apply_chat_template.return_value = Inputs(100)
        model = Mock()
        model.config.text_config.eos_token_id = 99
        model.config.text_end_token_id = 98
        model.generate.return_value = SimpleNamespace(sequences=Sequences())
        with self.assertRaisesRegex(ValueError, "before completing"):
            generate(
                {"mode": "transcribe"},
                model,
                processor,
                Mock(),
                SimpleNamespace(no_grad=nullcontext),
                "/tmp/wav",
            )
        processor.decode.assert_not_called()


class AudioOutputPrivacyTests(unittest.TestCase):
    def test_completed_asr_reasoning_is_removed_before_returning_transcript(self):
        self.assertEqual(
            public_text("<think>private</think>Hello<|im_end|>", transcribe=True),
            "Hello",
        )

    def test_spoken_reasoning_is_rejected_because_waveform_alignment_is_unverified(
        self,
    ):
        with self.assertRaisesRegex(ValueError, "reasoning in a spoken reply"):
            public_text(
                "<think>private</think>Hello<|audio_out|><|im_end|>", transcribe=False
            )

    def test_unfinished_reasoning_and_tool_markup_never_enter_call_history(self):
        for text in (
            "<think>private",
            "private</think>Hello",
            "<tool_call>run</tool_call>",
            "<function=run>",
            "<|im_start|>system",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                public_text(text, transcribe=False)

    def test_known_output_markers_are_removed_and_normal_text_is_preserved(self):
        self.assertEqual(
            public_text(
                "Hello < 3<|audio_out|> world<|text_end|><|im_end|>", transcribe=False
            ),
            "Hello < 3 world",
        )

    def test_output_tokens_are_reserved_inside_trial_context(self):
        self.assertEqual(MAX_INPUT_TOKENS + MAX_NEW_TOKENS, MAX_CONTEXT_TOKENS)


class AudioContextTests(unittest.TestCase):
    def test_released_maximum_context_includes_reserved_reply_at_exact_boundary(self):
        validate_context(32768 - MAX_NEW_TOKENS)
        with self.assertRaisesRegex(ValueError, "32768-token"):
            validate_context(32768 - MAX_NEW_TOKENS + 1)

    def test_history_has_no_extra_character_or_turn_cap_below_model_context(self):
        history = [
            {"role": "system", "text": "Instructions"},
            *({"role": "user", "text": "x" * 1500} for _ in range(16)),
        ]
        self.assertEqual(len(conversation({"mode": "reply", "messages": history})), 18)


class AudioReplyCompletionTests(unittest.TestCase):
    def fixture(self, ending):
        model = Mock()
        model.config.text_config.eos_token_id = 7
        model.config.text_end_token_id = 8
        codes = Mock()
        codes.__getitem__ = Mock(return_value="audio-code-slice")
        model.generate.return_value = SimpleNamespace(
            sequences=Sequences([ending]), audio_codes=codes, audio_lengths=[2]
        )
        processor = Mock()
        processor.apply_chat_template.return_value = Inputs(100)
        processor.decode.return_value = "Hello<|text_end|><|audio_out|><|im_end|>"
        return model, processor, Mock()

    def task(self):
        return {
            "mode": "reply",
            "messages": [
                {"role": "system", "text": "Persona"},
                {"role": "user", "text": "Hello"},
            ],
        }

    def test_mixed_text_end_rejects_partial_audio_before_codec_or_publication(self):
        model, processor, codec = self.fixture(8)
        with self.assertRaisesRegex(ValueError, "before completing"):
            generate(
                self.task(),
                model,
                processor,
                codec,
                SimpleNamespace(no_grad=nullcontext),
            )
        processor.decode.assert_not_called()
        codec.decode.assert_not_called()

    def test_actual_turn_eos_decodes_buffered_reply_and_encodes_24khz_pcm(self):
        model, processor, codec = self.fixture(7)
        waveform = Mock()
        samples = SimpleNamespace(size=2)
        waveform.float.return_value.cpu.return_value.numpy.return_value.reshape.return_value = samples
        codec.decode.return_value = SimpleNamespace(audio_values=[waveform])
        array = Mock()
        array.__mul__ = Mock(return_value=array)
        array.astype.return_value.tobytes.return_value = b"\x00\x00\xff\x7f"
        numpy = Mock()
        numpy.isfinite.return_value.all.return_value = True
        numpy.clip.return_value = array
        # Numeric conversion is a fake boundary here; no tensor, model or codec
        # execution occurs. The resulting WAV contract is inspected directly.
        with patch.dict("sys.modules", {"numpy": numpy}):
            result = generate(
                self.task(),
                model,
                processor,
                codec,
                SimpleNamespace(no_grad=nullcontext),
            )
        self.assertEqual(result["text"], "Hello")
        codec.decode.assert_called_once_with("audio-code-slice")
        with wave.open(
            io.BytesIO(base64.b64decode(result["audio_base64"])), "rb"
        ) as source:
            self.assertEqual(
                (
                    source.getframerate(),
                    source.getnchannels(),
                    source.getsampwidth(),
                    source.getnframes(),
                ),
                (24000, 1, 2, 2),
            )
            self.assertEqual(source.readframes(2), b"\x00\x00\xff\x7f")

    def test_transcription_text_end_remains_a_valid_text_only_terminal(self):
        model, processor, codec = self.fixture(8)
        processor.decode.return_value = "Hello<|text_end|>"
        result = generate(
            {"mode": "transcribe"},
            model,
            processor,
            codec,
            SimpleNamespace(no_grad=nullcontext),
            "/tmp/utterance.wav",
        )
        self.assertEqual(result, {"text": "Hello"})
        codec.decode.assert_not_called()
