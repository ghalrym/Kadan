import asyncio
import base64
import json
import struct
import unittest
import zlib
from unittest.mock import AsyncMock, MagicMock, patch

from api.pydantic_models.subscriptions import ActiveSubscription, CapabilityProviders
from api.routes.v1.images.generations import ImageRequest, create_image
from api.services.inference_errors import InferenceFailure
from api.services.subscription_images import (
    ImageSubscriptions,
    SubscriptionImages,
    decode_image,
)


def png():
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    return base64.b64encode(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\0\xff\0\0"))
        + chunk(b"IEND", b"")
    ).decode()


def events(count=1, item=None, status="completed"):
    result = [{"id": 1, "result": {}}, {"id": 2, "result": {"imageGeneration": True}}]
    for i in range(count):
        thread = f"thread-{i}"
        result += [
            {"id": 3 + i * 2, "result": {"thread": {"id": thread}}},
            {"id": 4 + i * 2, "result": {}},
            {
                "method": "item/completed",
                "params": {
                    "threadId": thread,
                    "item": item
                    or {
                        "type": "imageGeneration",
                        "status": "completed",
                        "result": png(),
                    },
                },
            },
            {
                "method": "turn/completed",
                "params": {
                    "threadId": thread,
                    "turn": {
                        "status": status,
                        "error": {"message": "provider failure"},
                    },
                },
            },
        ]
    return result


class Process:
    def __init__(self, values):
        self.values = values
        self.stdout = self
        self.stderr = MagicMock(read=AsyncMock(return_value=b""))
        self.stdin = MagicMock(drain=AsyncMock())
        self.returncode = None
        self.killed = False

    async def readline(self):
        if not self.values:
            await asyncio.Event().wait()
        return (json.dumps(self.values.pop(0)) + "\n").encode()

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        return self.returncode


class ImageTests(unittest.IsolatedAsyncioTestCase):
    def stream(self, **kwargs):
        return SubscriptionImages(
            ActiveSubscription(provider="codex", credential='{"tokens": {}}'),
            ImageRequest(count=1, prompt="a fox", **kwargs),
        )

    async def run_stream(self, values, stream=None):
        stream = stream or self.stream()
        process = Process(values)
        with patch(
            "api.services.subscription_images.asyncio.create_subprocess_exec",
            AsyncMock(return_value=process),
        ):
            result = await stream.generate()
        self.assertTrue(process.killed)
        self.assertIsNone(stream.home)
        return result, process

    async def test_completed_png_is_returned_without_inventing_seeds(self):
        result, process = await self.run_stream(events())
        self.assertEqual(result["image"].images_base64, [png()])
        self.assertEqual(result["image"].seeds, [])
        self.assertIn("1×1", result["image"].meta)
        commands = [
            json.loads(call.args[0]) for call in process.stdin.write.call_args_list
        ]
        self.assertIn("image_gen.imagegen", commands[3]["params"]["baseInstructions"])

    async def test_saved_path_without_artifact_data_is_not_success(self):
        with self.assertRaisesRegex(InferenceFailure, "invalid PNG"):
            await self.run_stream(
                events(
                    item={
                        "type": "imageGeneration",
                        "status": "completed",
                        "savedPath": "/tmp/art.png",
                        "result": "",
                    }
                )
            )

    async def test_image_event_before_rpc_response_is_retained(self):
        values = events()
        values[3], values[4] = values[4], values[3]
        result, _ = await self.run_stream(values)
        self.assertEqual(result["image"].images_base64, [png()])

    async def test_missing_cli_cleans_temporary_home(self):
        stream = self.stream()
        with (
            patch(
                "api.services.subscription_images.asyncio.create_subprocess_exec",
                AsyncMock(side_effect=FileNotFoundError),
            ),
            self.assertRaisesRegex(InferenceFailure, "unavailable"),
        ):
            await stream.generate()
        self.assertIsNone(stream.home)

    async def test_rpc_provider_error_is_reported(self):
        with self.assertRaisesRegex(InferenceFailure, "sign in again"):
            await self.run_stream([{"id": 1, "error": {"message": "sign in again"}}])

    async def test_count_uses_separate_generation_turns(self):
        stream = self.stream()
        stream.body.count = 4
        result, _ = await self.run_stream(events(4), stream)
        self.assertEqual(len(result["image"].images_base64), 4)

    async def test_failed_item_is_not_success_even_when_turn_completes(self):
        with self.assertRaisesRegex(InferenceFailure, "usage limit"):
            await self.run_stream(
                events(
                    item={
                        "type": "imageGeneration",
                        "status": "failed",
                        "failure": {"type": "usageLimitExceeded"},
                    }
                )
            )

    async def test_text_reply_does_not_substitute_for_artwork(self):
        with self.assertRaisesRegex(InferenceFailure, "exactly one"):
            await self.run_stream(events(item={"type": "agentMessage", "text": "done"}))

    async def test_failed_turn_does_not_publish_completed_artwork(self):
        with self.assertRaisesRegex(InferenceFailure, "provider failure"):
            await self.run_stream(events(status="failed"))

    async def test_unsupported_options_fail_before_start(self):
        for options in ({"seed": 0}, {"steps": 8}):
            stream = self.stream(**options)
            with patch.object(stream, "start", AsyncMock()) as start:
                with self.assertRaises(InferenceFailure) as error:
                    await stream.generate()
                self.assertEqual(error.exception.status_code, 422)
                start.assert_not_called()

    async def test_cancellation_reaps_process_and_removes_home(self):
        stream = self.stream()
        process = Process(events()[:4])
        with patch(
            "api.services.subscription_images.asyncio.create_subprocess_exec",
            AsyncMock(return_value=process),
        ):
            task = asyncio.create_task(stream.generate())
            for _ in range(10):
                await asyncio.sleep(0)
            home = stream.home
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(process.killed)
        self.assertFalse(home.exists())

    async def test_refreshed_sign_in_uses_existing_compare_and_swap(self):
        stream = self.stream()
        original = stream.start

        async def start():
            await original()
            (stream.home / "auth.json").write_text(
                '{"tokens": {"access_token": "renewed"}}'
            )

        with (
            patch.object(stream, "start", start),
            patch(
                "api.services.subscription_inference.subscriptions.refresh"
            ) as refresh,
        ):
            await self.run_stream(events(), stream)
        self.assertEqual(
            refresh.call_args.args[:2], ("codex", stream.subscription.credential)
        )

    async def test_missing_capability_fails_before_any_turn(self):
        with self.assertRaisesRegex(InferenceFailure, "unavailable"):
            await self.run_stream(
                [
                    {"id": 1, "result": {}},
                    {"id": 2, "result": {"imageGeneration": False}},
                ]
            )

    async def test_local_override_does_not_read_subscription_or_leak_to_worker(self):
        body = ImageRequest(prompt="fox", provider="local")
        self.assertNotIn("provider", body.model_dump(exclude_unset=True))

        async def infer(request, call):
            return await call

        with (
            patch("api.routes.v1.images.generations.infer", infer),
            patch(
                "api.routes.v1.images.generations.inference.submit",
                AsyncMock(return_value="local"),
            ) as submit,
            patch(
                "api.routes.v1.images.generations.image_subscriptions.generate",
                AsyncMock(),
            ) as subscription,
        ):
            self.assertEqual(await create_image(body, None), "local")
        submit.assert_awaited_once()
        subscription.assert_not_called()

    async def test_subscription_route_bypasses_native_admission(self):
        async def infer(request, call):
            return await call

        with (
            patch("api.routes.v1.images.generations.infer", infer),
            patch(
                "api.routes.v1.images.generations.inference.submit", AsyncMock()
            ) as submit,
            patch(
                "api.routes.v1.images.generations.image_subscriptions.generate",
                AsyncMock(return_value="subscription"),
            ),
        ):
            self.assertEqual(
                await create_image(ImageRequest(prompt="fox"), None), "subscription"
            )
        submit.assert_not_called()

    async def test_registry_cancellation_waits_for_cleanup(self):
        registry = ImageSubscriptions()
        cleaned = asyncio.Event()

        async def run():
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleaned.set()

        task = asyncio.create_task(run())
        await asyncio.sleep(0)
        registry.requests["job"] = task
        self.assertTrue(await registry.cancel("job"))
        self.assertTrue(cleaned.is_set())
        self.assertFalse(await registry.cancel("missing"))


class ImageDataTests(unittest.TestCase):
    def test_invalid_or_missing_artwork_is_rejected(self):
        for value in ("", "not base64", base64.b64encode(b"text").decode(), png()[:-8]):
            with self.assertRaises(InferenceFailure):
                decode_image(value)

    def test_image_selection_defaults_to_local_and_excludes_claude(self):
        self.assertEqual(CapabilityProviders().image, "local")
        with self.assertRaises(ValueError):
            CapabilityProviders(image="claude")
