"""Screenshot observations within the agent's existing computer scope."""

import base64

import httpx
from pydantic import Field

from ...api.config import ServiceError
from ...api.sandbox_client import MAX_RESULT_CHARACTERS, SandboxClient, truncated
from .base import Arguments, computer_tool


class LookArguments(Arguments):
    question: str = Field(
        min_length=1,
        description="What to inspect in the current computer screenshot.",
    )


@computer_tool("GET")
async def computer_look(args: LookArguments, computer: SandboxClient) -> dict:
    """Inspect the current computer screenshot with the selected vision provider.

    Ask about visible text, controls, positions or the current app state. The
    observation can be wrong; it does not execute actions or confirm later state.
    """
    if not args.question.strip():
        raise ServiceError("A screenshot question cannot be blank.", 422)
    image, media_type = await computer.screenshot()
    if len(image) > 2 * 1024**2 or media_type not in ("image/png", "image/jpeg"):
        raise ServiceError("Vision requires a PNG or JPEG screenshot up to 2 MiB.", 422)
    try:
        response = await computer.services.client.post(
            computer.services.settings.hikari_url + "/v1/vision/describe",
            json={
                "model": "lfm3-vl-2b-bf16",
                "image_base64": base64.b64encode(image).decode("ascii"),
                "media_type": media_type,
                "question": args.question,
            },
            timeout=httpx.Timeout(None, connect=3),
        )
    except httpx.RequestError as error:
        raise ServiceError("Hikari screenshot understanding is unavailable.") from error
    if not response.is_success:
        raise ServiceError(
            f"Hikari screenshot understanding returned HTTP {response.status_code}.",
            response.status_code,
            "vision_error",
        )
    try:
        result = response.json()
        text = result["text"]
        if (
            result["model"]
            not in ("lfm3-vl-2b-bf16", "claude-haiku-5-5", "gpt-6.1-terra")
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError("Invalid description")
    except (ValueError, KeyError, TypeError) as error:
        raise ServiceError(
            "Hikari returned an invalid screenshot description.", 502
        ) from error
    return {"model": result["model"], **truncated(text, MAX_RESULT_CHARACTERS)}


TOOLS = [computer_look]
