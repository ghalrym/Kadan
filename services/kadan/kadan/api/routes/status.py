from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ..pydantic_models.inference import DownloadState

router = APIRouter()


class ServiceState(BaseModel):
    state: Literal["ready", "unavailable"]
    message: str


class InferenceState(BaseModel):
    state: Literal["ready", "model_required", "unavailable"]
    message: str
    model_id: str | None = None
    download_state: DownloadState | None = None
    # The selected language model's context window in tokens, when Hikari reports it.
    context_limit: int | None = None
    # Whether the language model is resident; null when Hikari did not report it.
    model_state: (
        Literal["unloaded", "loading", "ready", "offloaded", "unloading", "error"]
        | None
    ) = None


class Status(BaseModel):
    tsuchi: ServiceState
    hikari: InferenceState


@router.get("/api/status")
async def status(request: Request) -> Status:
    desktop_ver = request.headers.get("x-kadan-desktop-version")
    if desktop_ver:
        can_update = (
            request.headers.get("x-kadan-desktop-can-update", "").strip().lower()
            == "true"
        )
        await request.app.state.services.updates.record_desktop_heartbeat(
            desktop_ver, can_update
        )
    return await request.app.state.services.status()
