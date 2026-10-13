import unittest

from pydantic import ValidationError

from api.routes.v1.chat.completions import CompletionRequest


class CompletionRequestTests(unittest.TestCase):
    def test_text_and_content_inputs_preserve_supported_message_fields(self):
        for field in ("text", "content"):
            with self.subTest(field=field):
                body = CompletionRequest(
                    messages=[{"role": "user", field: "Hello", "meta": "context"}]
                )
                self.assertEqual(
                    body.model_dump()["messages"],
                    [{"role": "user", "text": "Hello", "meta": "context"}],
                )

    def test_unsupported_message_fields_are_rejected_instead_of_discarded(self):
        for field, value in (
            (
                "tool_calls",
                [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": "{}"},
                    }
                ],
            ),
            ("function_call", {"name": "weather", "arguments": "{}"}),
            ("tool_call_id", "call_1"),
            ("name", "weather"),
            ("refusal", "I cannot help."),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValidationError) as error:
                    CompletionRequest(
                        messages=[
                            {"role": "assistant", "content": "Hello", field: value}
                        ]
                    )
                self.assertEqual(error.exception.errors()[0]["type"], "extra_forbidden")
                self.assertEqual(
                    error.exception.errors()[0]["loc"], ("messages", 0, field)
                )

    def test_ambiguous_text_and_content_are_rejected(self):
        with self.assertRaises(ValidationError):
            CompletionRequest(
                messages=[
                    {"role": "user", "text": "Hello", "content": "Different prompt"}
                ]
            )

    def test_unsupported_top_level_controls_are_rejected(self):
        for field, value in (
            ("tools", []),
            ("tool_choice", "auto"),
            ("reasoning_effort", "high"),
            ("stream_options", {"include_usage": True}),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValidationError) as error:
                    CompletionRequest(
                        messages=[{"role": "user", "content": "Hello"}],
                        **{field: value},
                    )
                self.assertEqual(error.exception.errors()[0]["type"], "extra_forbidden")
