"""Generate PNG artwork through Codex's built-in subscription image tool."""

import asyncio
import base64
import binascii
import json
import os
import struct
import tempfile
import zlib
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

import anyio
from api.pydantic_models.media import ImageSet
from api.pydantic_models.subscriptions import SUBSCRIPTION_MODELS, ActiveSubscription
from api.repositories import subscriptions
from api.services.inference_cancellation import await_cleanup
from api.services.inference_errors import InferenceFailure
from api.services.subscription_inference import SubscriptionStream


class SubscriptionImages(SubscriptionStream):
    def __init__(self, subscription: ActiveSubscription, body):
        # Reuse CLI lifecycle and compare-and-swap credential refresh, not chat prompting.
        self.subscription = subscription
        self.model = SUBSCRIPTION_MODELS["codex"]
        self.body = body
        self.home = None
        self.process = None
        self.stderr = None
        self.sequence = 0
        self.pending_events = []

    def _command(self):
        env = {**os.environ, "HOME": str(self.home), "CODEX_HOME": str(self.home)}
        env.pop("OPENAI_API_KEY", None)
        (self.home / "auth.json").write_text(self.subscription.credential)
        return [
            "codex",
            "app-server",
            "--listen",
            "stdio://",
            "-c",
            "features.image_generation=true",
            "-c",
            "features.shell_tool=false",
            "-c",
            "features.multi_agent=false",
            "-c",
            'web_search="disabled"',
        ], env

    async def start(self):
        # SubscriptionStream.start writes a chat prompt; app-server speaks JSON-RPC.
        self.home = Path(tempfile.mkdtemp(prefix="hikari-codex-images-"))
        command, env = self._command()
        try:
            self.process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.home,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=64 * 1024 * 1024,
            )
            self.stderr = asyncio.create_task(self.process.stderr.read())
        except FileNotFoundError as error:
            raise InferenceFailure(
                "ChatGPT images are unavailable in this Hikari installation."
            ) from error

    async def send(self, method, params, request_id=None):
        value = {"method": method, "params": params}
        if request_id is not None:
            value["id"] = request_id
        self.process.stdin.write((json.dumps(value) + "\n").encode())
        await self.process.stdin.drain()

    async def event(self):
        if self.pending_events:
            return self.pending_events.pop(0)
        return await self.read_event()

    async def read_event(self):
        line = await self.process.stdout.readline()
        if not line:
            raise InferenceFailure(
                "ChatGPT image request failed: " + await self._error_text(), 502
            )
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise TypeError("expected JSON object")
            return event
        except (ValueError, TypeError) as error:
            raise InferenceFailure(
                "ChatGPT returned an unreadable image response.", 502
            ) from error

    async def rpc(self, method, params):
        self.sequence += 1
        request_id = self.sequence
        await self.send(method, params, request_id)
        while True:
            event = await self.read_event()
            if event.get("id") == request_id:
                if "error" in event:
                    raise InferenceFailure(
                        "ChatGPT image request failed: "
                        + str(event["error"].get("message", event["error"]))[:2000],
                        502,
                    )
                return event["result"]
            self.reject_server_request(event)
            self.pending_events.append(event)

    def reject_server_request(self, event):
        if "method" in event and "id" in event:
            raise InferenceFailure(
                "ChatGPT requested an action this image request cannot perform.", 502
            )

    async def generate(self):
        if self.body.seed is not None:
            raise InferenceFailure(
                "ChatGPT images do not support seeds. Clear the seed or choose Local in Settings.",
                422,
            )
        if self.body.steps != 50:
            raise InferenceFailure(
                "ChatGPT images do not support sampling steps. Use the default or choose Local in Settings.",
                422,
            )
        try:
            await self.start()
            await self.rpc(
                "initialize",
                {
                    "clientInfo": {
                        "name": "hikari",
                        "title": "Hikari",
                        "version": "1.0",
                    },
                    "capabilities": {"experimentalApi": True},
                },
            )
            await self.send("initialized", {})
            capabilities = await self.rpc("modelProvider/capabilities/read", {})
            if not capabilities.get("imageGeneration"):
                raise InferenceFailure(
                    "Image generation is unavailable with this ChatGPT connection. Choose Local in Settings."
                )
            images, sizes = [], []
            for _ in range(self.body.count):
                thread = await self.rpc(
                    "thread/start",
                    {
                        "model": self.model,
                        "cwd": str(self.home),
                        "ephemeral": True,
                        "approvalPolicy": "never",
                        "sandbox": "read-only",
                        "baseInstructions": "Generate artwork using only the built-in image_gen.imagegen tool. Do not use other tools. Generate exactly one image. Do not edit an earlier image. Return after the image is generated.",
                    },
                )
                thread_id = thread["thread"]["id"]
                await self.rpc(
                    "turn/start",
                    {
                        "threadId": thread_id,
                        "input": [
                            {
                                "type": "text",
                                "text": f"Create one new image. Requested aspect ratio: {self.body.aspect}.\n\n{self.body.prompt}",
                            }
                        ],
                    },
                )
                artwork = []
                while True:
                    event = await self.event()
                    self.reject_server_request(event)
                    params = event.get("params") or {}
                    if params.get("threadId") != thread_id:
                        continue
                    if event.get("method") == "item/completed":
                        item = params.get("item") or {}
                        if item.get("type") == "imageGeneration":
                            if item.get("status") != "completed" or item.get("failure"):
                                failure = item.get("failure") or {}
                                message = (
                                    "Your ChatGPT image usage limit has been reached. Wait for it to reset or choose Local in Settings."
                                    if failure.get("type") == "usageLimitExceeded"
                                    else "ChatGPT could not generate this image. Try again or check your connection in Settings."
                                )
                                raise InferenceFailure(message, 502)
                            artwork.append(decode_image(item.get("result", "")))
                    elif event.get("method") == "turn/completed":
                        turn = params["turn"]
                        if turn.get("status") != "completed":
                            raise InferenceFailure(
                                "ChatGPT image generation failed: "
                                + str(
                                    (turn.get("error") or {}).get("message")
                                    or turn.get("status")
                                )[:2000],
                                502,
                            )
                        break
                    elif event.get("method") == "error" and not params.get("willRetry"):
                        raise InferenceFailure(
                            "ChatGPT image generation failed: "
                            + str(params.get("error", {}))[:2000],
                            502,
                        )
                if len(artwork) != 1:
                    raise InferenceFailure(
                        "ChatGPT did not return exactly one completed image. Try again.",
                        502,
                    )
                encoded, size = artwork[0]
                images.append(encoded)
                sizes.append(size)
            return {
                "image": ImageSet(
                    id=uuid4().hex,
                    mode="Generate",
                    prompt=self.body.prompt,
                    aspect={
                        "1:1": "square",
                        "4:3": "landscape",
                        "3:4": "portrait",
                        "16:9": "wide",
                    }[self.body.aspect],
                    seeds=[],
                    meta="ChatGPT · "
                    + ", ".join(sizes)
                    + " · requested aspect "
                    + self.body.aspect,
                    images_base64=images,
                    mime_type="image/png",
                )
            }
        except (OSError, ValueError) as error:
            raise InferenceFailure(
                "ChatGPT image connection failed. Try again or reconnect in Settings.",
                502,
            ) from error
        finally:
            with anyio.CancelScope(shield=True):
                await self.aclose()


def decode_image(result: str):
    """Only publish actual PNG bytes from the pinned protocol's image result."""
    try:
        raw = base64.b64decode(result, validate=True)
        if len(raw) < 33 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
            raise ValueError("not PNG")
        width, height = struct.unpack(">II", raw[16:24])
        if not width or not height:
            raise ValueError("invalid dimensions")
        offset, has_data, ended = 8, False, False
        while offset + 12 <= len(raw):
            length = struct.unpack(">I", raw[offset : offset + 4])[0]
            end = offset + 12 + length
            if end > len(raw):
                raise ValueError("truncated chunk")
            kind = raw[offset + 4 : offset + 8]
            expected_crc = struct.unpack(">I", raw[end - 4 : end])[0]
            if zlib.crc32(raw[offset + 4 : end - 4]) != expected_crc:
                raise ValueError("corrupt PNG chunk")
            has_data |= kind == b"IDAT"
            offset = end
            if kind == b"IEND":
                ended = length == 0 and offset == len(raw)
                break
        if not has_data or not ended:
            raise ValueError("incomplete PNG")
    except (ValueError, binascii.Error, TypeError) as error:
        raise InferenceFailure(
            "ChatGPT returned missing or invalid PNG artwork.", 502
        ) from error
    return base64.b64encode(raw).decode(), f"{width}×{height}"


class ImageSubscriptions:
    def __init__(self):
        self.requests = {}

    async def generate(self, body, job_id):
        try:
            subscription = await asyncio.to_thread(subscriptions.active, "image")
        except subscriptions.SubscriptionError as error:
            raise InferenceFailure(str(error), 409) from error
        if subscription is None:
            return None
        if subscription.provider != "codex":
            raise InferenceFailure("This subscription cannot generate images.")
        if job_id in self.requests:
            raise InferenceFailure("This image request is already active.", 409)
        task = asyncio.current_task()
        self.requests[job_id] = task
        try:
            return await SubscriptionImages(subscription, body).generate()
        finally:
            self.requests.pop(job_id, None)

    async def cancel(self, job_id):
        task = self.requests.get(job_id)
        if task is None:
            return False
        task.cancel()
        with suppress(asyncio.CancelledError, InferenceFailure):
            await await_cleanup(task)
        return True

    async def close(self):
        for job_id in tuple(self.requests):
            await self.cancel(job_id)


image_subscriptions = ImageSubscriptions()
