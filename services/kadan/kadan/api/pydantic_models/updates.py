from pydantic import BaseModel, Field


class DesktopClientStatus(BaseModel):
    connected: bool = False
    current_version: str | None = None
    update_available: bool = False
    can_self_update: bool = False


class PendingDesktopUpdate(BaseModel):
    request_id: str
    target_version: str
    requested_at: int


class UpdateCheckResult(BaseModel):
    current_version: str
    latest_version: str | None = None
    update_available: bool = False
    release_tag: str | None = None
    release_url: str | None = None
    release_notes: str | None = None
    published_at: str | None = None
    can_self_update: bool = False
    install_kind: str = "manual"
    reason: str | None = None
    staged_version: str | None = None
    desktop: DesktopClientStatus = Field(default_factory=DesktopClientStatus)


class UpdateStatusResponse(BaseModel):
    state: str = "idle"
    current_version: str
    staged_version: str | None = None
    message: str | None = None
    desktop: DesktopClientStatus = Field(default_factory=DesktopClientStatus)
    pending_desktop_update: PendingDesktopUpdate | None = None


class StageUpdateRequest(BaseModel):
    target_version: str | None = None


class StageUpdateResponse(BaseModel):
    ok: bool = True
    target_version: str
    server_staged: bool = False
    desktop_update_available: bool = False
    message: str


class ApplyUpdateRequest(BaseModel):
    target_version: str | None = None
    include_desktop: bool = True


class ApplyUpdateResponse(BaseModel):
    ok: bool = True
    target_version: str
    server_restarting: bool = False
    desktop_update_requested: bool = False
    pending_desktop_update: PendingDesktopUpdate | None = None
    message: str


class DesktopAckRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=128)
