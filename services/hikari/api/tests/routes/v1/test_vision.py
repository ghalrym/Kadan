"""Route logic with queue boundaries replaced; no server or inference."""

import base64
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException
from pydantic import ValidationError

from api.routes.v1 import vision


def request(**changes):
    return vision.DescribeRequest(
        **{
            "image_base64": base64.b64encode(b"\x89PNG\r\n\x1a\nunit payload").decode(),
            "media_type": "image/png",
            "question": "What is visible?",
            **changes,
        }
    )


class VisionRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_wrong_declared_format_fails_before_queue_or_inference(self):
        for media_type, content in (
            ("image/png", b"\xff\xd8\xff\xe1jpeg payload"),
            ("image/jpeg", b"\x89PNG\r\n\x1a\npng payload"),
        ):
            with self.subTest(media_type=media_type):
                submit = Mock(return_value=object())
                infer = AsyncMock(
                    return_value={"model": "lfm3-vl-2b-bf16", "text": "Unused"}
                )
                with (
                    patch.object(vision.inference, "submit", submit),
                    patch.object(vision, "infer", infer),
                    self.assertRaises(HTTPException) as error,
                ):
                    await vision.describe_screenshot(
                        request(
                            media_type=media_type,
                            image_base64=base64.b64encode(content).decode(),
                        ),
                        object(),
                    )
                self.assertEqual(error.exception.status_code, 422)
                self.assertIn("signature", error.exception.detail)
                submit.assert_not_called()
                infer.assert_not_awaited()

    async def test_invalid_encoded_screenshot_never_enters_queue(self):
        submit = AsyncMock()
        with (
            patch.object(vision.inference, "submit", submit),
            self.assertRaises(HTTPException) as error,
        ):
            await vision.describe_screenshot(
                request(image_base64="not base64!"), object()
            )
        self.assertEqual(error.exception.status_code, 422)
        submit.assert_not_called()

    async def test_matching_formats_keep_queue_routing_and_result_validation(self):
        for media_type, content in (
            ("image/png", b"\x89PNG\r\n\x1a\nunit payload"),
            ("image/jpeg", b"\xff\xd8\xff\xe1unit payload"),
        ):
            with self.subTest(media_type=media_type):
                body = request(
                    media_type=media_type,
                    image_base64=base64.b64encode(content).decode(),
                )
                submit = AsyncMock(
                    return_value={
                        "model": "lfm3-vl-2b-bf16",
                        "text": "A dialog is visible.",
                    }
                )

                async def await_queue(request, call):
                    return await call

                with (
                    patch.object(vision.inference, "submit", submit),
                    patch.object(vision, "infer", await_queue),
                ):
                    response = await vision.describe_screenshot(body, object())
                    submit.assert_awaited_once_with(body, feature="vision")
                    queued = submit.call_args.args[0]
                    self.assertEqual(
                        queued.model_dump(exclude_unset=True)["model"],
                        "lfm3-vl-2b-bf16",
                    )
                    self.assertEqual(queued.image_base64, body.image_base64)
                    self.assertEqual(response.text, "A dialog is visible.")
                    submit.return_value = {
                        "model": "different-model",
                        "text": "Wrong source",
                    }
                    with self.assertRaises(HTTPException) as error:
                        await vision.describe_screenshot(body, object())
                self.assertEqual(error.exception.status_code, 502)

    def test_request_defaults_to_model_remaining_context_and_large_output_is_accepted(
        self,
    ):
        self.assertIsNone(request().max_new_tokens)
        self.assertEqual(request(question="x" * 5000).question, "x" * 5000)
        self.assertEqual(request(max_new_tokens=16384).max_new_tokens, 16384)
        self.assertEqual(
            len(
                vision.DescribeResponse(model="lfm3-vl-2b-bf16", text="x" * 40000).text
            ),
            40000,
        )

    def test_request_rejects_blank_question(self):
        with self.assertRaises(ValidationError):
            request(question=" ")

    def test_request_rejects_excessive_output_tokens(self):
        with self.assertRaises(ValidationError):
            request(max_new_tokens=128000)

    def test_request_rejects_boolean_output_tokens(self):
        with self.assertRaises(ValidationError):
            request(max_new_tokens=True)

    def test_request_rejects_external_image_url(self):
        with self.assertRaises(ValidationError):
            request(image_url="https://example.com/image.png")
