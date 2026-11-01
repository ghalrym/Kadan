"""Provider image transport and lifecycle with subprocess boundaries mocked."""

import asyncio
import base64
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from api.pydantic_models.subscriptions import ActiveSubscription
from api.routes.v1.vision import DescribeRequest
from api.services.inference_errors import InferenceFailure
from api.services.subscription_vision import SubscriptionVision
from api.tests.services.test_subscription_inference import FakeProcess


def request(**changes):
    data = io.BytesIO()
    Image.new("RGB", (2, 2)).save(data, format="PNG")
    return DescribeRequest(
        image_base64=base64.b64encode(data.getvalue()).decode(),
        media_type="image/png",
        question="What is visible?",
        **changes,
    )


class VisionSubscriptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_claude_receives_image_block_and_question(self):
        body = request()
        stream = SubscriptionVision(
            ActiveSubscription(provider="claude", credential="token"), body
        )
        process = FakeProcess(
            [
                json.dumps(
                    {
                        "type": "stream_event",
                        "event": {"delta": {"type": "text_delta", "text": "A window"}},
                    }
                ),
                json.dumps({"type": "result", "is_error": False}),
            ]
        )
        with patch(
            "api.services.subscription_inference.asyncio.create_subprocess_exec",
            return_value=process,
        ) as spawn:
            self.assertEqual(await stream.text(), "A window")
        self.assertIn("--input-format", spawn.call_args.args)
        content = json.loads(process.stdin.data)["message"]["content"]
        self.assertEqual(content[0]["source"]["data"], body.image_base64)
        self.assertEqual(content[1]["text"], body.question)
        self.assertFalse(Path(spawn.call_args.kwargs["cwd"]).exists())

    async def test_codex_attaches_bytes_and_cleans_up_after_refresh(self):
        body = request()
        stream = SubscriptionVision(
            ActiveSubscription(provider="codex", credential='{"tokens":{}}'), body
        )
        process = FakeProcess(
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "A window"},
                    }
                ),
                json.dumps({"type": "turn.completed"}),
            ]
        )

        async def spawn(*command, **kwargs):
            self.image = Path(command[command.index("--image") + 1])
            self.assertEqual(
                self.image.read_bytes(), base64.b64decode(body.image_base64)
            )
            (self.image.parent / "auth.json").write_text(
                '{"tokens":{"access_token":"new"}}'
            )
            return process

        with (
            patch(
                "api.services.subscription_inference.asyncio.create_subprocess_exec",
                side_effect=spawn,
            ),
            patch(
                "api.services.subscription_inference.subscriptions.refresh"
            ) as refresh,
        ):
            self.assertEqual(await stream.text(), "A window")
        refresh.assert_called_once()
        self.assertNotIn(str(self.image).encode(), process.stdin.data)
        self.assertFalse(self.image.parent.exists())

    async def test_cancellation_waits_for_process_and_removes_image(self):
        stream = SubscriptionVision(
            ActiveSubscription(provider="codex", credential='{"tokens":{}}'), request()
        )
        process = FakeProcess([])

        async def cancelled():
            raise asyncio.CancelledError

        process.stdin.drain = cancelled
        with (
            patch(
                "api.services.subscription_inference.asyncio.create_subprocess_exec",
                return_value=process,
            ) as spawn,
            self.assertRaises(asyncio.CancelledError),
        ):
            await stream.text()
        self.assertTrue(process.killed)
        self.assertIsNotNone(process.returncode)
        self.assertFalse(Path(spawn.call_args.kwargs["cwd"]).exists())

    def test_invalid_image_and_unhonored_budget_are_rejected(self):
        subscription = ActiveSubscription(provider="claude", credential="token")
        for body in (
            request(max_new_tokens=20),
            request().model_copy(update={"image_base64": "YWJj"}),
            request().model_copy(update={"media_type": "image/jpeg"}),
        ):
            with self.assertRaises(InferenceFailure) as error:
                SubscriptionVision(subscription, body)
            self.assertEqual(error.exception.status_code, 422)

    async def test_provider_failure_cleans_up_without_returning_an_observation(self):
        for provider, event in (
            ("claude", {"type": "result", "is_error": True, "result": "usage limit"}),
            ("codex", {"type": "turn.failed", "error": {"message": "usage limit"}}),
        ):
            with self.subTest(provider=provider):
                stream = SubscriptionVision(
                    ActiveSubscription(provider=provider, credential='{"tokens":{}}'),
                    request(),
                )
                process = FakeProcess([json.dumps(event)], returncode=1)
                with (
                    patch(
                        "api.services.subscription_inference.asyncio.create_subprocess_exec",
                        return_value=process,
                    ) as spawn,
                    self.assertRaisesRegex(InferenceFailure, "usage limit"),
                ):
                    await stream.text()
                self.assertIsNotNone(process.returncode)
                self.assertFalse(Path(spawn.call_args.kwargs["cwd"]).exists())
