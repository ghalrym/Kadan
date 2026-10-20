from ipaddress import ip_address
from typing import Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class ServiceState(BaseModel):
    state: Literal["ready", "unavailable"]
    message: str
    browser_url: str


class InferenceState(BaseModel):
    state: Literal["ready", "tier_required", "model_required", "unavailable"]
    message: str
    tier: str | None = None
    model_id: str | None = None
    # The selected language model's context window in tokens, when Hikari reports it.
    context_limit: int | None = None
    # Whether the language model is resident; null when Hikari did not report it.
    model_state: (
        Literal["unloaded", "loading", "ready", "offloaded", "unloading", "error"]
        | None
    ) = None
    settings_url: str


class Status(BaseModel):
    tsuchi: ServiceState
    hikari: InferenceState


def browser_link(url: str, hostname: str) -> str:
    """Use the validated browser host for local links; retain explicit remote URLs."""
    parsed = urlsplit(url)
    local = parsed.hostname == "localhost"
    try:
        local = local or ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        pass
    if not local:
        return url
    host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
    return parsed._replace(netloc=netloc).geturl()


@router.get("/api/status")
async def status(request: Request) -> Status:
    result = await request.app.state.services.status()
    # LocalRequestSafety validates Host before routing. Do not trust forwarded hosts.
    hostname = request.url.hostname
    assert hostname is not None
    return result.model_copy(
        update={
            "tsuchi": result.tsuchi.model_copy(
                update={
                    "browser_url": browser_link(result.tsuchi.browser_url, hostname),
                }
            ),
            "hikari": result.hikari.model_copy(
                update={
                    "settings_url": browser_link(result.hikari.settings_url, hostname),
                }
            ),
        }
    )
