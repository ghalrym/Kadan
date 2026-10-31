"""Pure formatting and validation tests; never load or run a model."""

import unittest

from types import SimpleNamespace
from unittest.mock import Mock

from experimental_worker.text import (
    CONTEXT_LIMIT,
    MODEL_ID,
    ReasoningBudget,
    execute,
    finish_reason,
    generate_text,
    messages,
    reply_text,
    output_budget,
)


class TextTests(unittest.TestCase):
    def test_full_context_prefill_is_delegated_without_slicing_or_duplicate_decode(
        self,
    ):
        tokens, mask = object(), object()
        model = SimpleNamespace(generate=Mock(return_value="output"))
        self.assertEqual(
            generate_text(model, {"input_ids": tokens, "attention_mask": mask}, 254208),
            "output",
        )
        model.generate.assert_called_once()
        options = model.generate.call_args.kwargs
        self.assertIs(options["input_ids"], tokens)
        self.assertIs(options["attention_mask"], mask)
        self.assertEqual(options["prefill_chunk_size"], 512)
        self.assertEqual(options["cache_implementation"], "dynamic")
        self.assertTrue(options["use_cache"])
        self.assertTrue(options["disable_compile"])
        self.assertEqual(options["logits_to_keep"], 1)
        self.assertEqual(options["max_new_tokens"], 254208)

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
            "not finished, with the opening marker in the prompt",
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
            reply_text("private</think><function=finish></function><|im_end|>\n"),
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

    def test_generation_can_use_all_context_after_the_complete_prompt(self):
        self.assertEqual(output_budget(7936), 254208)
        self.assertEqual(output_budget(CONTEXT_LIMIT - 1), 1)
        self.assertEqual(output_budget(10, 32768), 32758)
        for tokens in (CONTEXT_LIMIT, CONTEXT_LIMIT + 1):
            with (
                self.subTest(tokens=tokens),
                self.assertRaisesRegex(ValueError, "not truncated"),
            ):
                output_budget(tokens)

    def test_output_is_capped_by_the_requested_limit(self):
        self.assertEqual(output_budget(10, 32768, 1000), 1000)
        self.assertEqual(output_budget(32000, 32768, 1000), 768)

    def test_reasoning_is_closed_at_its_budget_then_left_alone(self):
        end = 7
        budget = ReasoningBudget(prompt_tokens=3, budget=2, end_id=end)
        scores = unittest.mock.MagicMock()
        budget(self.token_ids([1, 2, 3, 4]), scores)
        scores.__setitem__.assert_not_called()
        budget(self.token_ids([1, 2, 3, 4, 5]), scores)
        self.assertEqual(
            scores.__setitem__.call_args_list,
            [
                unittest.mock.call((slice(None), slice(None)), float("-inf")),
                unittest.mock.call((slice(None), end), 0),
            ],
        )
        scores.reset_mock()
        budget(self.token_ids([1, 2, 3, 4, 5, end]), scores)
        budget(self.token_ids([1, 2, 3, 4, 5, end, 8, 9]), scores)
        scores.__setitem__.assert_not_called()

    def test_reasoning_closed_early_is_never_forced(self):
        budget = ReasoningBudget(prompt_tokens=3, budget=2, end_id=7)
        scores = unittest.mock.MagicMock()
        budget(self.token_ids([1, 2, 3, 7]), scores)
        budget(self.token_ids([1, 2, 3, 7, 4, 5]), scores)
        scores.__setitem__.assert_not_called()

    @staticmethod
    def token_ids(tokens):
        tensor = unittest.mock.MagicMock()
        tensor.shape = (1, len(tokens))
        tensor.__getitem__.return_value = tokens[-1]
        return tensor

    def test_invalid_token_limits_are_rejected_before_dependencies(self):
        for field in ("max_tokens", "reasoning_tokens"):
            for value in (0, True, "10"):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaisesRegex(ValueError, "token limits"),
                ):
                    execute(
                        {
                            "model": MODEL_ID,
                            "messages": [{"role": "user", "text": "Hello"}],
                            field: value,
                        },
                        "completion",
                        "/unused",
                        lambda _: None,
                    )

    def test_eos_at_the_output_boundary_is_complete(self):
        self.assertEqual(finish_reason([4, 124900], 124900), "stop")
        self.assertEqual(finish_reason([4, 124900], [124900, 124901]), "stop")
        for tokens in ([4, 5], [], [4, 124899]):
            with self.subTest(tokens=tokens):
                self.assertEqual(finish_reason(tokens, 124900), "length")
