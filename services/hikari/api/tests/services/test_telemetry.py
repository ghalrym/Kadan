import json
import unittest
from typing import get_args

from api.pydantic_models.inference_task import LlmModel
from api.services.telemetry import MAX_BODY_SUMMARY_BYTES, summarize


class RequestSummaryTests(unittest.TestCase):
    def summary(self, value):
        body = json.dumps(value, ensure_ascii=False).encode()
        return summarize(body, len(body), True)

    def test_each_valid_message_alias_counts_characters_without_retaining_text(self):
        for field in ("text", "content"):
            with self.subTest(field=field):
                summary, model = self.summary(
                    {"messages": [{"role": "user", field: "Private message"}]}
                )
                self.assertIn("1 messages; 15 message characters", summary)
                self.assertNotIn("Private message", summary)
                self.assertIsNone(model)

    def test_mixed_aliases_count_unicode_characters_once(self):
        summary, _ = self.summary(
            {
                "messages": [
                    {"role": "user", "text": "Hello"},
                    {"role": "assistant", "content": "世界🌙"},
                    {"role": "user", "content": "e\u0301"},
                ]
            }
        )
        self.assertIn("3 messages; 10 message characters", summary)
        self.assertNotIn("世界", summary)

    def test_invalid_message_entries_and_nonstring_aliases_are_safe(self):
        summary, _ = self.summary(
            {
                "messages": [
                    None,
                    "private scalar",
                    [],
                    {},
                    {"text": None},
                    {"text": 123},
                    {"content": ["private list"]},
                    {"content": {"private": "object"}},
                    {"content": "ok"},
                ]
            }
        )
        self.assertIn("9 messages; 2 message characters", summary)
        self.assertNotIn("private", summary)

    def test_ambiguous_aliases_rejected_by_chat_do_not_contribute_text(self):
        summary, _ = self.summary(
            {"messages": [{"text": "Private text", "content": "Private content"}]}
        )
        self.assertIn("1 messages; 0 message characters", summary)
        self.assertNotIn("Private", summary)

    def test_malformed_and_nonobject_json_remain_structural(self):
        for body in (b"{private", b"\xff", b"[1, 2]"):
            with self.subTest(body=body):
                summary, model = summarize(body, len(body), True)
                self.assertNotIn("private", summary)
                self.assertIsNone(model)
        self.assertEqual(
            summarize(b"{private", 8, True),
            ("8 request bytes; non-JSON or malformed JSON", None),
        )

    def test_summary_keeps_curated_models_but_never_private_fields(self):
        public_model = get_args(LlmModel)[0]
        for model in (public_model, "Private model"):
            with self.subTest(model=model):
                summary, saved_model = self.summary(
                    {
                        "messages": [{"role": "user", "content": "Private message"}],
                        "model": model,
                        "credentials": "Private token",
                        "audio": "Private media",
                    }
                )
                self.assertIn("15 message characters", summary)
                self.assertNotIn("Private", summary)
                self.assertEqual(
                    saved_model, public_model if model == public_model else None
                )

    def test_existing_byte_cap_and_incomplete_body_rules_are_unchanged(self):
        self.assertEqual(MAX_BODY_SUMMARY_BYTES, 16_384)
        prefix = b'{"messages":[{"content":"secret"}],"padding":"'
        body = prefix + b"x" * (MAX_BODY_SUMMARY_BYTES - len(prefix) - 2) + b'"}'
        self.assertEqual(len(body), MAX_BODY_SUMMARY_BYTES)
        summary, model = summarize(body, len(body), True)
        self.assertIn("1 messages; 6 message characters", summary)
        self.assertNotIn("secret", summary)
        self.assertIsNone(model)
        self.assertEqual(
            summarize(body, len(body) + 1, True),
            ("16385 request bytes; payload exceeds summary limit", None),
        )
        self.assertEqual(
            summarize(body, len(body), False),
            ("16384 request bytes received; body incomplete or not consumed", None),
        )
