from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ..pydantic_models.inference import DownloadState

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
    download_state: DownloadState | None = None
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


@router.get("/api/status")
async def status(request: Request) -> Status:
    return await request.app.state.services.status()
