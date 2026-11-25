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


ALLOWED_VISION_MODELS = (
    "inkling-2-mini-iq4-nl",
    "inkling-2-mini-q8-0",
    "claude-haiku-5-5",
    "gpt-6.1-terra",
)


@computer_tool("GET")
async def computer_look(args: LookArguments, computer: SandboxClient) -> dict:
    """Inspect the current computer screenshot with the active language model.

    Ask about visible text, controls, positions or the current app state. The
    observation can be wrong; it does not execute actions or confirm later state.
    """
    if not args.question.strip():
        raise ServiceError("A screenshot question cannot be blank.", 422)
    image, media_type = await computer.screenshot()
    if len(image) > 2 * 1024**2 or media_type not in ("image/png", "image/jpeg"):
        raise ServiceError("Vision requires a PNG or JPEG screenshot up to 2 MiB.", 422)
    image_url = f"data:{media_type};base64,{base64.b64encode(image).decode('ascii')}"
    try:
        response = await computer.services.client.post(
            computer.services.settings.hikari_url + "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": args.question},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                "stream": False,
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
        model = result["model"]
        text = result["choices"][0]["message"]["content"]
        if (
            model not in ALLOWED_VISION_MODELS
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError("Invalid description")
    except (ValueError, KeyError, TypeError, IndexError) as error:
        raise ServiceError(
            "Hikari returned an invalid screenshot description.", 502
        ) from error
    return {"model": model, **truncated(text, MAX_RESULT_CHARACTERS)}


TOOLS = [computer_look]
