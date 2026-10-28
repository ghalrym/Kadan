import base64
import io
import unittest
import wave
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from experimental_worker.audio import (
    MAX_CONTEXT_TOKENS,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    audio_bytes,
    conversation,
    generate,
    public_text,
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
    def __getitem__(self, item):
        return [7]


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
        with self.assertRaisesRegex(ValueError, "8192-token"):
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
