"""Route logic with queue boundaries replaced; no server or inference."""

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from pydantic import ValidationError

from api.routes.v1 import vision


def request(**changes):
    return vision.DescribeRequest(
        **{
            "image_base64": "YWJj",
            "media_type": "image/png",
            "question": "What is visible?",
            **changes,
        }
    )


class VisionRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_encoded_screenshot_never_enters_queue(self):
        submit = AsyncMock()
        with patch.object(vision.inference, "submit", submit):
            with self.assertRaises(HTTPException) as error:
                await vision.describe_screenshot(
                    request(image_base64="not base64!"), object()
                )
        self.assertEqual(error.exception.status_code, 422)
        submit.assert_not_called()

    async def test_valid_request_uses_vision_queue_and_validates_worker_result(self):
        body = request()
        submit = AsyncMock(
            return_value={"model": "lfm3-vl-2b-bf16", "text": "A dialog is visible."}
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
                queued.model_dump(exclude_unset=True)["model"], "lfm3-vl-2b-bf16"
            )
            self.assertEqual(response.text, "A dialog is visible.")
            submit.return_value = {"model": "different-model", "text": "Wrong source"}
            with self.assertRaises(HTTPException) as error:
                await vision.describe_screenshot(body, object())
        self.assertEqual(error.exception.status_code, 502)

    def test_request_rejects_blank_question(self):
        with self.assertRaises(ValidationError):
            request(question=" ")

    def test_request_rejects_excessive_output_tokens(self):
        with self.assertRaises(ValidationError):
            request(max_new_tokens=1025)

    def test_request_rejects_boolean_output_tokens(self):
        with self.assertRaises(ValidationError):
            request(max_new_tokens=True)

    def test_request_rejects_external_image_url(self):
        with self.assertRaises(ValidationError):
            request(image_url="https://example.com/image.png")
