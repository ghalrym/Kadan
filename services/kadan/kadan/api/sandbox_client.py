"""The agent's computer: an AIO sandbox reached only through its own HTTP API.

The agent's tools in `kadan/runner/tools/` call its routes through `run`.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx

from .config import ServiceError

if TYPE_CHECKING:
    from .services import LocalServices

# Results enter the agent's bounded transcript, so they are truncated here.
MAX_RESULT_CHARACTERS = 4000
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_SCREENSHOT_BYTES = 16 * 1024 * 1024
OPERATION_SECONDS = 300
# Event streams never end on their own; return what arrived in this window.
STREAM_SECONDS = 10
TEXT_TYPES = ("application/json", "text/", "application/xml", "application/javascript")
# The sandbox's page automation can lose its browser session and hang; its gateway
# then answers 504 after 60 seconds until the browser is restarted.
BROWSER_RESTART_PATH = "/v1/browser/restart"
BROWSER_RESTART_SECONDS = 60
BROWSER_RECOVERED = (
    "The browser stopped responding, so Kadan restarted it and retried. "
    "Open tabs and page state were reset."
)


def truncated(value: str, limit: int) -> dict:
    if len(value) <= limit:
        return {"body": value}
    return {"body": value[:limit], "truncated": True, "characters": len(value)}


class SandboxClient:
    def __init__(self, services: LocalServices):
        self.services = services

    @property
    def configured(self) -> bool:
        return self.services.settings.sandbox_url is not None

    @property
    def base_url(self) -> str:
        url = self.services.settings.sandbox_url
        if url is None:
            raise ServiceError(
                "The agent's computer is not configured. Set KADAN_SANDBOX_URL.",
                503,
                "computer_not_configured",
            )
        return url

    @property
    def api_key(self) -> str | None:
        key = self.services.settings.sandbox_api_key
        return (key.get_secret_value() or None) if key else None

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def vnc_url(self) -> str:
        """The sandbox's websockify endpoint; Kadan relays it with the API key."""
        return "ws" + self.base_url.removeprefix("http") + "/websockify"

    async def request(
        self, method: str, path: str, seconds: float = 15, **kwargs
    ) -> httpx.Response:
        try:
            response = await self.services.client.request(
                method,
                self.base_url + path,
                headers=self.headers(),
                timeout=httpx.Timeout(seconds, connect=3),
                **kwargs,
            )
        except httpx.RequestError as error:
            raise ServiceError(
                "The agent's computer is unavailable. Check that kadan-sandbox is running."
            ) from error
        if not response.is_success:
            raise ServiceError(
                f"The agent's computer returned HTTP {response.status_code}.",
                502 if response.status_code < 500 else 503,
                "computer_error",
            )
        return response

    async def context(self) -> dict:
        try:
            value = (await self.request("GET", "/v1/sandbox")).json()
        except ValueError as error:
            raise ServiceError(
                "The agent's computer returned an invalid response.",
                502,
                "invalid_response",
            ) from error
        return value.get("data", value) if isinstance(value, dict) else {}

    async def screenshot(self) -> tuple[bytes, str]:
        response = await self.request(
            "GET", "/v1/browser/screenshot", params={"format": "jpeg", "quality": 70}
        )
        media_type = response.headers.get("content-type", "")
        if not media_type.startswith("image/") or len(response.content) > (
            MAX_SCREENSHOT_BYTES
        ):
            raise ServiceError(
                "The agent's computer returned an invalid screenshot.",
                502,
                "invalid_response",
            )
        return response.content, media_type.split(";")[0]

    async def run(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        data: dict[str, str] | None = None,
        files: dict[str, tuple[str, bytes]] | None = None,
    ) -> dict:
        """Call one sandbox route for a tool and summarize the response for the agent.

        Failed HTTP statuses are returned, not raised, so the agent can correct
        itself. A jammed browser is restarted once and the call retried.
        """
        request: dict[str, Any] = {"params": params, "data": data, "files": files}
        if json_body is not None:
            request["json"] = json_body
        result = await self.send(method, path, request)
        if (
            path.startswith("/v1/browser/")
            and path != BROWSER_RESTART_PATH
            and result["status"] == 504
        ):
            try:
                await self.request(
                    "POST", BROWSER_RESTART_PATH, seconds=BROWSER_RESTART_SECONDS
                )
            except ServiceError:
                return {
                    **result,
                    "error": "The browser stopped responding and could not be restarted.",
                }
            result = await self.send(method, path, request)
            result["recovered"] = BROWSER_RECOVERED
        return result

    async def send(self, method: str, path: str, request: dict[str, Any]) -> dict:
        try:
            async with asyncio.timeout(OPERATION_SECONDS):
                async with self.services.client.stream(
                    method,
                    self.base_url + path,
                    headers=self.headers(),
                    timeout=httpx.Timeout(OPERATION_SECONDS, connect=3),
                    **request,
                ) as response:
                    media_type = response.headers.get("content-type", "")
                    content, complete = await self.read(
                        response, media_type.startswith("text/event-stream")
                    )
        except (httpx.RequestError, TimeoutError) as error:
            raise ServiceError(
                f"The agent's computer did not complete {method} {path}.",
                503,
                "computer_unavailable",
            ) from error
        result: dict[str, Any] = {
            "status": response.status_code,
            "content_type": media_type.split(";")[0],
        }
        if not media_type.startswith(TEXT_TYPES):
            return {
                **result,
                "bytes": len(content),
                "note": "Binary content is not shown to the agent.",
            }
        text = content.decode(errors="replace")
        if media_type.startswith("application/json") and complete:
            try:
                text = json.dumps(json.loads(text), separators=(",", ":"))
                if len(text) <= MAX_RESULT_CHARACTERS:
                    return {**result, "body": json.loads(text)}
            except ValueError:
                pass
        return {
            **result,
            **truncated(text, MAX_RESULT_CHARACTERS),
            **({} if complete else {"incomplete": True}),
        }

    @staticmethod
    async def read(response: httpx.Response, stream: bool) -> tuple[bytes, bool]:
        content = bytearray()
        try:
            async with asyncio.timeout(STREAM_SECONDS if stream else None):
                async for chunk in response.aiter_bytes():
                    content.extend(chunk)
                    if len(content) > MAX_RESPONSE_BYTES:
                        return bytes(content[:MAX_RESPONSE_BYTES]), False
        except TimeoutError:
            return bytes(content), False
        return bytes(content), True


def segment(value: object) -> str:
    """A path parameter, encoded so it stays one segment of the sandbox route."""
    return quote(str(value), safe="")
