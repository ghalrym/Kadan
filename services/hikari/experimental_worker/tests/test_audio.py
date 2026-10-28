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
    execute,
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


class FusedAudioExecutorTests(unittest.TestCase):
    def execute(self, generate_results, *, probe=False, sentence=False):
        torch = Mock()
        torch.cuda.is_available.return_value = True
        torch.cuda.is_bf16_supported.return_value = True
        model = Mock()
        model.eval.return_value = model
        processor = Mock()
        codec = Mock()
        processor.audio_tokenizer.to.return_value.eval.return_value = codec
        transformers = SimpleNamespace(
            AutoModelForMultimodalLM=Mock(), AutoProcessor=Mock()
        )
        transformers.AutoModelForMultimodalLM.from_pretrained.return_value = model
        transformers.AutoProcessor.from_pretrained.return_value = processor
        calls = []

        def generated(task, *args, **kwargs):
            calls.append((task, args, kwargs))
            result = generate_results[len(calls) - 1]
            if isinstance(result, Exception):
                raise result
            return dict(result)

        events = []
        task = {
            "feature": "audio_call",
            "model_id": "lfm3-audio-2b-bf16",
            "mode": "turn",
            "codec_streaming_probe": probe,
            "sentence_tts": sentence,
            "audio": wav_uri(),
            "messages": [
                {"role": "system", "text": "Persona"},
                *[{"role": "user", "text": "x" * 1000} for _ in range(12)],
            ],
        }
        with (
            patch.dict("sys.modules", {"torch": torch, "transformers": transformers}),
            patch("experimental_worker.audio.generate", side_effect=generated),
        ):
            try:
                result = execute(
                    task,
                    "generate",
                    "/tmp/checkpoint",
                    lambda event: events.append((event, len(calls))),
                )
            except ValueError as error:
                result = error
        return result, calls, events, transformers, model, processor, codec

    def test_single_model_load_runs_two_fresh_mock_generations_and_emits_asr_first(
        self,
    ):
        result, calls, events, transformers, model, processor, codec = self.execute(
            [{"text": "Hi"}, {"text": "Hello", "audio_base64": "AAAA"}]
        )
        self.assertEqual(result["transcript"], "Hi")
        self.assertEqual([call[0]["mode"] for call in calls], ["transcribe", "reply"])
        transformers.AutoModelForMultimodalLM.from_pretrained.assert_called_once()
        transformers.AutoProcessor.from_pretrained.assert_called_once()
        self.assertTrue(all(call[1][:3] == (model, processor, codec) for call in calls))
        self.assertEqual(
            events, [({"state": "audio_transcript", "transcript": "Hi"}, 1)]
        )
        self.assertEqual(len(calls[1][0]["messages"]), 13)
        self.assertEqual(calls[1][0]["messages"][-1], {"role": "user", "text": "Hi"})
        self.assertNotIn("past_key_values", calls[1][0])

    def test_failed_reply_returns_no_audio_and_does_not_erase_completed_transcript(
        self,
    ):
        result, calls, events, *_ = self.execute(
            [{"text": "Hi"}, ValueError("before completing the reply")]
        )
        self.assertIsInstance(result, ValueError)
        self.assertEqual(
            events, [({"state": "audio_transcript", "transcript": "Hi"}, 1)]
        )
        self.assertEqual(len(calls), 2)

    def test_empty_transcript_skips_reply_generation(self):
        result, calls, _events, *_ = self.execute([{"text": ""}])
        self.assertEqual(result["transcript"], "")
        self.assertIsNone(result["audio_base64"])
        self.assertEqual(len(calls), 1)

    def test_probe_is_passed_only_to_reply_and_numeric_diagnostics_survive_setup_timing(
        self,
    ):
        result, calls, *_ = self.execute(
            [
                {"text": "Hi"},
                {
                    "text": "Hello",
                    "audio_base64": "AAAA",
                    "timing": {"streamed_chunk_count": 2},
                },
            ],
            probe=True,
        )
        self.assertEqual(calls[0][2], {})
        self.assertIs(calls[1][2]["codec_streaming_probe"], True)
        self.assertTrue(callable(calls[1][2]["emit"]))
        self.assertEqual(result["timing"]["streamed_chunk_count"], 2)
        self.assertIn("model_load_seconds", result["timing"])

    def test_sentence_turn_loads_once_and_runs_asr_answer_then_known_scripts(self):
        result, calls, events, transformers, model, processor, codec = self.execute(
            [
                {"text": "Hi"},
                {"text": "First. Second."},
                {"script": "First.", "sample_count": 2400, "audio_base64": "AAAA"},
                {"script": "Second.", "sample_count": 2400, "audio_base64": "BBBB"},
            ],
            sentence=True,
        )
        self.assertEqual(
            [entry[0]["mode"] for entry in calls],
            ["transcribe", "answer", "tts", "tts"],
        )
        self.assertTrue(
            all(entry[1][:3] == (model, processor, codec) for entry in calls)
        )
        transformers.AutoModelForMultimodalLM.from_pretrained.assert_called_once()
        transformers.AutoProcessor.from_pretrained.assert_called_once()
        self.assertEqual(
            [entry[0]["state"] for entry in events],
            ["audio_transcript", "audio_answer", "audio_segment", "audio_segment"],
        )
        self.assertEqual(result["scripts"], ["First.", "Second."])
        self.assertEqual(result["timing"]["tts_segment_count"], 2)
