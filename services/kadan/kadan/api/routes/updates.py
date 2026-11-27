from fastapi import APIRouter, Request

from ..pydantic_models.updates import (
    ApplyUpdateRequest,
    ApplyUpdateResponse,
    DesktopAckRequest,
    StageUpdateRequest,
    StageUpdateResponse,
    UpdateCheckResult,
    UpdateStatusResponse,
)

router = APIRouter()


async def _record_desktop_headers(request: Request) -> None:
    version = request.headers.get("x-kadan-desktop-version")
    if not version:
        return
    can_update = (
        request.headers.get("x-kadan-desktop-can-update", "").strip().lower() == "true"
    )
    await request.app.state.services.updates.record_desktop_heartbeat(
        version, can_update
    )


@router.get("/api/updates")
async def check_updates(request: Request) -> UpdateCheckResult:
    await _record_desktop_headers(request)
    return await request.app.state.services.updates.check()


@router.get("/api/updates/status")
async def update_status(request: Request) -> UpdateStatusResponse:
    await _record_desktop_headers(request)
    return await request.app.state.services.updates.status()


@router.post("/api/updates/desktop/ack")
async def ack_desktop_update(
    payload: DesktopAckRequest, request: Request
) -> dict[str, bool]:
    acknowledged = await request.app.state.services.updates.ack_desktop_update(
        payload.request_id
    )
    return {"ok": True, "acknowledged": acknowledged}


@router.post("/api/updates/stage")
async def stage_update(
    request: Request, payload: StageUpdateRequest | None = None
) -> StageUpdateResponse:
    await _record_desktop_headers(request)
    target = payload.target_version if payload else None
    return await request.app.state.services.updates.stage(target_version=target)


@router.post("/api/updates/apply")
async def apply_update(
    request: Request, payload: ApplyUpdateRequest | None = None
) -> ApplyUpdateResponse:
    await _record_desktop_headers(request)
    target = payload.target_version if payload else None
    include_desktop = payload.include_desktop if payload else True
    return await request.app.state.services.updates.apply(
        target_version=target, include_desktop=include_desktop
    )
