"""Pure input/metadata tests; no model import, loading or inference."""

import base64
import unittest
from unittest.mock import patch

from experimental_worker.vision import (
    MAX_CONTEXT,
    MODEL,
    description,
    execute,
    image_bytes,
    output_budget,
    validate_completion,
    validate_context,
)

from api.pydantic_models.inference_task import InferenceTask


class VisionWorkerTests(unittest.TestCase):
    def test_context_uses_exact_checkpoint_maximum_and_reserves_output_without_truncation(
        self,
    ):
        self.assertEqual(MAX_CONTEXT, 128000)
        self.assertEqual(MODEL["context_limit"], MAX_CONTEXT)
        self.assertEqual(MODEL["architecture_context_limit"], MAX_CONTEXT)
        validate_context(8000, 1024)
        validate_context(126976, 1024)
        with self.assertRaisesRegex(ValueError, "no input was truncated"):
            validate_context(126977, 1024)

    def test_omitted_budget_uses_every_remaining_token_and_explicit_caps_can_exceed_1024(
        self,
    ):
        self.assertEqual(output_budget(100), 127900)
        self.assertEqual(output_budget(100, 16384), 16384)
        self.assertEqual(output_budget(127999), 1)
        with self.assertRaises(ValueError):
            output_budget(128000)
        with self.assertRaises(ValueError):
            output_budget(100, 127901)

    def test_eos_at_the_exact_budget_is_complete_and_short_non_eos_is_not(self):
        validate_completion(127900, 7, [7, 8])
        for count, token in ((0, None), (2, 99)):
            with self.assertRaisesRegex(ValueError, "terminal EOS"):
                validate_completion(count, token, 7)

    def test_context_rejects_empty_prompt(self):
        with self.assertRaisesRegex(ValueError, "must contain tokens"):
            validate_context(0, 1024)

    def test_context_rejects_empty_reply(self):
        with self.assertRaisesRegex(ValueError, "must contain tokens"):
            validate_context(1, 0)

    def test_context_rejects_negative_prompt_tokens(self):
        with self.assertRaisesRegex(ValueError, "must contain tokens"):
            validate_context(-1, 1024)

    def test_binary_screenshot_is_recovered_without_interpreting_paths_or_urls(self):
        for media_type, content in (
            ("image/png", b"\x89PNG\r\n\x1a\nunit payload"),
            ("image/jpeg", b"\xff\xd8\xff\xe1unit payload"),
        ):
            with self.subTest(media_type=media_type):
                self.assertEqual(
                    image_bytes(
                        {
                            "media_type": media_type,
                            "image_base64": base64.b64encode(content).decode(),
                        }
                    ),
                    content,
                )
        for value in ("https://example.com/screenshot.png", "/tmp/screenshot.png", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                image_bytes({"media_type": "image/png", "image_base64": value})

    def test_mismatched_or_truncated_signatures_are_rejected(self):
        for media_type, content in (
            ("image/png", b"\xff\xd8\xff\xe1jpeg payload"),
            ("image/jpeg", b"\x89PNG\r\n\x1a\npng payload"),
            ("image/png", b"\x89PNG"),
            ("image/jpeg", b"\xff\xd8"),
            ("image/png", b"abc"),
        ):
            with (
                self.subTest(media_type=media_type, content=content),
                self.assertRaisesRegex(ValueError, "signature"),
            ):
                image_bytes(
                    {
                        "media_type": media_type,
                        "image_base64": base64.b64encode(content).decode(),
                    }
                )

    def test_matching_signature_keeps_the_exact_compressed_byte_limit(self):
        limit = 2 * 1024**2
        prefix = b"\x89PNG\r\n\x1a\n"
        for size in (limit, limit + 1):
            content = prefix + b"x" * (size - len(prefix))
            task = {
                "media_type": "image/png",
                "image_base64": base64.b64encode(content).decode(),
            }
            with self.subTest(size=size):
                if size == limit:
                    self.assertEqual(image_bytes(task), content)
                else:
                    with self.assertRaisesRegex(ValueError, "2 MiB"):
                        image_bytes(task)

    def test_screenshot_payload_is_bounded_and_supported_media_only(self):
        for task in (
            {"media_type": "image/gif", "image_base64": "YWJj"},
            {"media_type": "image/png", "image_base64": "A" * 2_796_205},
            {
                "media_type": "image/jpeg",
                "image_base64": base64.b64encode(b"a" * (2 * 1024**2 + 1)).decode(),
            },
        ):
            with (
                self.subTest(
                    media_type=task["media_type"],
                    encoded_bytes=len(task["image_base64"]),
                ),
                self.assertRaises(ValueError),
            ):
                image_bytes(task)

    def test_vision_is_its_own_task_and_cannot_replace_image_generation(self):
        task = InferenceTask(
            input={
                "feature": "vision",
                "media_type": "image/png",
                "image_base64": "YWJj",
                "question": "What is visible?",
            }
        )
        self.assertEqual(task.feature, "vision")
        self.assertEqual(task.input.model, "lfm3-vl-2b-bf16")
        self.assertEqual(
            InferenceTask.model_validate_json(task.model_dump_json()), task
        )

    def assert_invalid_work(self, **changes):
        with self.assertRaises(ValueError):
            execute(
                {
                    "question": "Describe",
                    "media_type": "image/png",
                    "image_base64": "YWJj",
                    **changes,
                },
                "generate",
                "/unused",
                lambda event: None,
            )

    def test_blank_question_is_rejected_before_loading_runtime_dependencies(self):
        self.assert_invalid_work(question=" ")

    def test_wrong_image_signature_never_imports_inference_dependencies(self):
        with (
            patch("builtins.__import__") as imports,
            self.assertRaisesRegex(ValueError, "signature"),
        ):
            execute(
                {
                    "question": "Describe",
                    "media_type": "image/png",
                    "image_base64": base64.b64encode(b"\xff\xd8\xffjpeg").decode(),
                },
                "generate",
                "/unused",
                lambda event: None,
            )
        imports.assert_not_called()

    def test_excessive_output_is_rejected_before_loading_runtime_dependencies(self):
        self.assert_invalid_work(max_new_tokens=128000)

    def test_boolean_output_is_rejected_before_loading_runtime_dependencies(self):
        self.assert_invalid_work(max_new_tokens=True)

    def test_unknown_model_is_rejected_before_loading_runtime_dependencies(self):
        self.assert_invalid_work(model="another-model")

    def test_final_description_omits_reasoning_and_requires_a_completed_answer(self):
        for text in (
            "<think>internal</think>Visible button",
            "internal</think>Visible button<|im_end|>",
            "Visible button<|im_end|><|endoftext|>",
        ):
            with self.subTest(text=text):
                self.assertEqual(description(text), "Visible button")
        for text in ("", "<think>unfinished", "<think>only reasoning</think>"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                description(text)
        with self.assertRaisesRegex(ValueError, "another conversation turn"):
            description("Visible button<|im_start|>user")

    def test_pinned_manifest_has_weights_custom_code_and_license(self):
        self.assertEqual(len(MODEL["revision"]), 40)
        self.assertEqual(MODEL["kind"], "vision")
        files = {item["name"]: item for item in MODEL["artifacts"]}
        self.assertTrue(
            {
                "LICENSE",
                "model.safetensors",
                "modeling_lfm3_vl.py",
                "processing_lfm3_vl.py",
            }
            <= files.keys()
        )
        self.assertEqual(
            MODEL["estimated_bytes"], sum(item["size"] for item in files.values())
        )
        self.assertEqual(files["model.safetensors"]["algorithm"], "sha256")
        self.assertTrue(
            all(MODEL["revision"] in item["url"] for item in files.values())
        )
