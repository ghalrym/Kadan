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
        with (
            patch.object(vision.subscriptions, "active", return_value=None),
            patch.object(vision.inference, "submit", submit),
            self.assertRaises(HTTPException) as error,
        ):
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
            patch.object(vision.subscriptions, "active", return_value=None),
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
            submit.return_value = {"model": "gpt-6.1-terra", "text": "Wrong source"}
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

    async def test_subscription_vision_bypasses_local_queue(self):
        from api.pydantic_models.subscriptions import ActiveSubscription

        stream = AsyncMock()
        stream.model = "gpt-6.1-terra"
        stream.text.return_value = "A browser window."

        async def await_call(request, call):
            return await call

        with (
            patch.object(
                vision.subscriptions,
                "active",
                return_value=ActiveSubscription(
                    provider="codex", credential='{"tokens":{}}'
                ),
            ) as active,
            patch.object(vision, "SubscriptionVision", return_value=stream),
            patch.object(vision, "infer", await_call),
            patch.object(vision.inference, "submit", AsyncMock()) as submit,
        ):
            result = await vision.describe_screenshot(request(), object())
        active.assert_called_once_with("vision")
        submit.assert_not_called()
        self.assertEqual(result.model, "gpt-6.1-terra")
        self.assertEqual(result.text, "A browser window.")

    async def test_missing_selected_credential_fails_without_local_fallback(self):
        with (
            patch.object(
                vision.subscriptions,
                "active",
                side_effect=vision.subscriptions.SubscriptionError("Sign in again"),
            ),
            patch.object(vision.inference, "submit", AsyncMock()) as submit,
            self.assertRaises(HTTPException) as error,
        ):
            await vision.describe_screenshot(request(), object())
        self.assertEqual(error.exception.status_code, 409)
        submit.assert_not_called()
