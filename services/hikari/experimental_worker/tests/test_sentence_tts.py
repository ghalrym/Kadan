import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from experimental_worker.audio import conversation, generate, sentence_reply
from experimental_worker.speech_segments import speech_scripts
from experimental_worker.tests import test_audio as audio_tests

TORCH = SimpleNamespace(no_grad=nullcontext, cuda=SimpleNamespace(synchronize=Mock()))
MESSAGES = [{"role": "system", "text": "Saved persona"}, {"role": "user", "text": "Hi"}]


class SentenceTtsTests(unittest.TestCase):
    def test_shared_boundaries_cover_sentences_whitespace_and_unbroken_words(self):
        self.assertEqual(
            speech_scripts("First. Second!\nThird?  "), ["First.", "Second!", "Third?"]
        )
        self.assertEqual(
            [len(script) for script in speech_scripts("x" * 500)], [240, 240, 20]
        )
        self.assertEqual(speech_scripts("   "), [])

    def test_official_tts_prompt_contains_only_known_script(self):
        prompt = conversation({"mode": "tts", "script": "Known words."})
        self.assertEqual(
            prompt,
            [
                {
                    "role": "system",
                    "content": "Perform TTS in American English using a female voice.",
                },
                {"role": "user", "content": "Known words."},
                {"role": "assistant", "content": "<|audio_start|>"},
            ],
        )
        answer = conversation({"mode": "answer", "messages": MESSAGES})
        self.assertEqual(answer[0]["content"], "Saved persona")
        self.assertEqual(answer[-2], {"role": "user", "content": "Hi"})
        self.assertEqual(
            answer[-1], {"role": "assistant", "content": "<think></think>"}
        )

    def test_public_answer_forces_text_only_and_strips_completed_reasoning_before_speech(
        self,
    ):
        for decoded in ("Public answer", "<think>private</think>Public answer"):
            model, processor, codec = audio_tests.AudioReplyCompletionTests().fixture(7)
            processor.decode.return_value = decoded
            self.assertEqual(
                generate(
                    {"mode": "answer", "messages": MESSAGES},
                    model,
                    processor,
                    codec,
                    TORCH,
                ),
                {"text": "Public answer"},
            )
            self.assertEqual(model.generate.call_args.kwargs["generation_mode"], "text")
            self.assertTrue(model.generate.call_args.kwargs["suppress_mixed"])
            self.assertTrue(model.generate.call_args.kwargs["suppress_audio_out"])
            self.assertTrue(
                processor.apply_chat_template.call_args.kwargs["continue_final_message"]
            )
            self.assertNotIn(
                "add_generation_prompt", processor.apply_chat_template.call_args.kwargs
            )
            codec.decode.assert_not_called()
        model, processor, codec = audio_tests.AudioReplyCompletionTests().fixture(7)
        processor.decode.return_value = "<think>unfinished"
        with self.assertRaises(ValueError):
            generate(
                {"mode": "answer", "messages": MESSAGES}, model, processor, codec, TORCH
            )
        codec.decode.assert_not_called()

    def test_invalid_answer_emits_no_segments_or_public_answer(self):
        emit = Mock()
        with (
            patch(
                "experimental_worker.audio.generate",
                side_effect=ValueError("unfinished answer"),
            ) as generated,
            self.assertRaises(ValueError),
        ):
            sentence_reply(MESSAGES, Mock(), Mock(), Mock(), TORCH, emit)
        generated.assert_called_once()
        emit.assert_not_called()

    def test_complete_first_segment_is_emitted_before_later_generation_failure(self):
        emit = Mock()
        with (
            patch(
                "experimental_worker.audio.generate",
                side_effect=[
                    {"text": "First. Second."},
                    {"script": "First.", "sample_count": 2400, "audio_base64": "AAAA"},
                    ValueError("incomplete later TTS"),
                ],
            ) as generated,
            self.assertRaisesRegex(ValueError, "later TTS"),
        ):
            sentence_reply(MESSAGES, Mock(), Mock(), Mock(), TORCH, emit)
        self.assertEqual(
            [entry.args[0]["state"] for entry in emit.call_args_list],
            ["audio_answer", "audio_segment"],
        )
        self.assertEqual(
            emit.call_args_list[1].args[0]["audio_segment"]["script"], "First."
        )
        self.assertEqual(
            [entry.args[0]["mode"] for entry in generated.call_args_list],
            ["answer", "tts", "tts"],
        )

    def test_cumulative_audio_limit_fails_instead_of_truncating_second_segment(self):
        emit = Mock()
        with (
            patch(
                "experimental_worker.audio.generate",
                side_effect=[
                    {"text": "First. Second."},
                    {
                        "script": "First.",
                        "sample_count": 60 * 24000,
                        "audio_base64": "AAAA",
                    },
                    {"script": "Second.", "sample_count": 1, "audio_base64": "BBBB"},
                ],
            ),
            self.assertRaisesRegex(ValueError, "sixty seconds"),
        ):
            sentence_reply(MESSAGES, Mock(), Mock(), Mock(), TORCH, emit)
        self.assertEqual(len(emit.call_args_list), 2)

    def test_final_receipt_lists_the_known_scripts_and_numeric_timing(self):
        emit = Mock()
        with patch(
            "experimental_worker.audio.generate",
            side_effect=[
                {"text": "First. Second."},
                {"script": "First.", "sample_count": 2400, "audio_base64": "AAAA"},
                {"script": "Second.", "sample_count": 2400, "audio_base64": "BBBB"},
            ],
        ):
            result = sentence_reply(MESSAGES, Mock(), Mock(), Mock(), TORCH, emit)
        self.assertEqual(result["scripts"], ["First.", "Second."])
        self.assertEqual(result["segment_count"], 2)
        self.assertEqual(
            [
                entry.args[0]["audio_segment"]["index"]
                for entry in emit.call_args_list[1:]
            ],
            [0, 1],
        )
        self.assertEqual(result["timing"]["tts_sample_count"], 4800)
        self.assertGreaterEqual(result["timing"]["first_tts_segment_seconds"], 0)

    def test_tts_valid_noninterleaved_text_end_returns_known_script_and_pcm(self):
        model, processor, codec = audio_tests.AudioReplyCompletionTests().fixture(8)
        processor.decode.return_value = "<|audio_out|><|text_end|>"
        waveform, array = Mock(), Mock()
        waveform.float.return_value.cpu.return_value.numpy.return_value.reshape.return_value = SimpleNamespace(
            size=2
        )
        codec.decode.return_value = SimpleNamespace(audio_values=[waveform])
        array.__mul__ = Mock(return_value=array)
        array.astype.return_value.tobytes.return_value = b"\x00\x00\xff\x7f"
        numpy = Mock()
        numpy.isfinite.return_value.all.return_value = True
        numpy.clip.return_value = array
        with patch.dict("sys.modules", {"numpy": numpy}):
            result = generate(
                {"mode": "tts", "script": "Known words."},
                model,
                processor,
                codec,
                TORCH,
            )
        self.assertEqual(result["script"], "Known words.")
        self.assertEqual(result["sample_count"], 2)
        self.assertTrue(result["audio_base64"])
        self.assertNotIn("generation_mode", model.generate.call_args.kwargs)
        self.assertTrue(
            processor.apply_chat_template.call_args.kwargs["continue_final_message"]
        )

    def test_tts_cap_visible_text_and_private_markers_reject_before_decoding(self):
        for ending, decoded in (
            (99, ""),
            (7, "Unexpected words"),
            (7, "<think>private</think>"),
            (7, "<tool_call>run</tool_call>"),
            (8, "<|mixed_start|><|audio_out|><|text_end|>"),
            (8, "<|audio_start|><|audio_out|><|text_end|>"),
            (8, "<|mixed_end|><|audio_out|><|text_end|>"),
        ):
            model, processor, codec = audio_tests.AudioReplyCompletionTests().fixture(
                ending
            )
            processor.decode.return_value = decoded
            with (
                self.subTest(ending=ending, decoded=decoded),
                self.assertRaises(ValueError),
            ):
                generate(
                    {"mode": "tts", "script": "Known words."},
                    model,
                    processor,
                    codec,
                    TORCH,
                )
            codec.decode.assert_not_called()
