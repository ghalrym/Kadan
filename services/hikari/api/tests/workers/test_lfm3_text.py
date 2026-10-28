"""Pure formatting and validation tests; never load or run a model."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from experimental_worker.text import (
    CONTEXT_LIMIT,
    MAX_NEW_TOKENS,
    MODEL_ID,
    execute,
    generate_text,
    messages,
    reply_text,
    validate_context,
)


class TextTests(unittest.TestCase):
    def test_full_context_prefill_is_delegated_without_slicing_or_duplicate_decode(self):
        tokens, mask = object(), object()
        model = SimpleNamespace(generate=Mock(return_value="output"))
        self.assertEqual(generate_text(model, {"input_ids": tokens, "attention_mask": mask}), "output")
        model.generate.assert_called_once()
        options = model.generate.call_args.kwargs
        self.assertIs(options["input_ids"], tokens)
        self.assertIs(options["attention_mask"], mask)
        self.assertEqual(options["prefill_chunk_size"], 512)
        self.assertEqual(options["cache_implementation"], "dynamic")
        self.assertTrue(options["use_cache"])
        self.assertTrue(options["disable_compile"])
        self.assertEqual(options["logits_to_keep"], 1)

    def test_preserves_existing_rendered_tool_history(self):
        text = "<tool_response>\nSaved\n</tool_response>"
        self.assertEqual(
            messages({"messages": [{"role": "user", "text": text}]}),
            [{"role": "user", "content": text}],
        )

    def test_rejects_unsupported_history_without_inference(self):
        for rows in (
            [],
            [{"role": "tool", "text": "x"}],
            [{"role": "user", "text": ""}],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                messages({"messages": rows})

    def test_private_reasoning_is_removed_before_tool_parsing(self):
        self.assertEqual(
            reply_text("<think>private</think>\n<tool_call>call</tool_call>"),
            "<tool_call>call</tool_call>",
        )

    def test_unfinished_reasoning_is_an_error(self):
        for text in (
            "<think>not finished",
            "<think>closed</think>answer<think>unfinished",
        ):
            with (
                self.subTest(text=text),
                self.assertRaisesRegex(ValueError, "unfinished"),
            ):
                reply_text(text)

    def test_terminal_markers_do_not_change_tool_xml(self):
        self.assertEqual(
            reply_text("<function=finish></function><|im_end|>\n"),
            "<function=finish></function>",
        )

    def test_native_model_is_never_substituted(self):
        with self.assertRaisesRegex(ValueError, "only the explicit"):
            execute(
                {"model": "native", "messages": []},
                "completion",
                "/unused",
                lambda _: None,
            )

    def test_manual_load_does_not_claim_residency(self):
        with self.assertRaisesRegex(ValueError, "generation only"):
            execute({"model": MODEL_ID}, "load", "/unused", lambda _: None)

    def test_invalid_context_is_rejected_before_dependencies(self):
        for limit in (0, False, CONTEXT_LIMIT + 1):
            with (
                self.subTest(limit=limit),
                self.assertRaisesRegex(ValueError, "context"),
            ):
                execute(
                    {
                        "model": MODEL_ID,
                        "messages": [{"role": "user", "text": "Hello"}],
                        "context_limit": limit,
                    },
                    "completion",
                    "/unused",
                    lambda _: None,
                )

    def test_full_supported_context_boundary_reserves_reply(self):
        validate_context(CONTEXT_LIMIT - MAX_NEW_TOKENS)
        with self.assertRaisesRegex(ValueError, "not truncated"):
            validate_context(CONTEXT_LIMIT - MAX_NEW_TOKENS + 1)
