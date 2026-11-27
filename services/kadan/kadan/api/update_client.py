"""Update checking and coordination with Kadan Server's local control endpoint."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import uuid4

import httpx

from .config import ServiceError
from .pydantic_models.updates import (
    ApplyUpdateResponse,
    DesktopClientStatus,
    PendingDesktopUpdate,
    StageUpdateResponse,
    UpdateCheckResult,
    UpdateStatusResponse,
)

if TYPE_CHECKING:
    from .services import LocalServices

DESKTOP_STALE_SECONDS = 120


def parse_version(raw: str) -> tuple[int, int, int, int] | None:
    trimmed = raw.strip().lstrip("vV")
    if not trimmed:
        return None
    core, _, suffix = trimmed.partition("-")
    parts = core.split(".")
    if len(parts) > 3:
        return None
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
    except ValueError:
        return None
    revision = 0
    if suffix:
        digits = "".join(ch for ch in suffix if ch.isdigit())
        if digits:
            revision = int(digits)
    return (major, minor, patch, revision)


def is_newer_version(current: str, latest: str) -> bool:
    cur = parse_version(current)
    lat = parse_version(latest)
    if cur is not None and lat is not None:
        return lat > cur
    c = current.strip().lstrip("vV")
    l = latest.strip().lstrip("vV")
    return bool(l) and l != c


class UpdateClient:
    """Checks for Kadan updates and delegates installs to Kadan Server when available."""

    def __init__(self, services: LocalServices):
        self.services = services
        # Used only when running outside Kadan Server (e.g. Docker / dev).
        self._desktop_version: str | None = None
        self._desktop_can_update: bool = False
        self._desktop_seen_at: float = 0.0
        self._latest_known: str | None = None
        self._pending_desktop: PendingDesktopUpdate | None = None

    def _control_url(self) -> str | None:
        return getattr(self.services.settings, "server_control_url", None)

    def _control_headers(self) -> dict[str, str]:
        token = getattr(self.services.settings, "server_control_token", None)
        if token:
            return {"Authorization": f"Bearer {token}"}
        return {}

    async def _control_request(
        self, method: str, path: str, *, timeout: float = 30, json: dict | None = None
    ) -> dict:
        base = self._control_url()
        assert base is not None
        try:
            response = await self.services.client.request(
                method,
                base + path,
                headers=self._control_headers(),
                json=json,
                timeout=httpx.Timeout(timeout, connect=3),
            )
        except httpx.RequestError as error:
            raise ServiceError(
                "Kadan Server's update service is currently unavailable.",
                503,
                "updater_unavailable",
            ) from error
        if not response.is_success:
            try:
                detail = response.json().get("detail")
            except (ValueError, AttributeError):
                detail = None
            if not isinstance(detail, str):
                detail = f"Kadan Server updater returned HTTP {response.status_code}."
            raise ServiceError(detail, response.status_code, "update_error")
        return response.json()

    def _fallback_desktop_status(
        self, latest: str | None = None
    ) -> DesktopClientStatus:
        if not self._desktop_version:
            return DesktopClientStatus()
        connected = (time.time() - self._desktop_seen_at) <= DESKTOP_STALE_SECONDS
        ref_latest = latest or self._latest_known
        update_available = (
            connected
            and bool(ref_latest)
            and is_newer_version(self._desktop_version, ref_latest)
        )
        return DesktopClientStatus(
            connected=connected,
            current_version=self._desktop_version,
            update_available=update_available,
            can_self_update=self._desktop_can_update,
        )

    async def record_desktop_heartbeat(
        self, version: str | None, can_self_update: bool
    ) -> None:
        if not version or not version.strip():
            return
        clean = version.strip()
        if self._control_url():
            try:
                await self._control_request(
                    "POST",
                    "/updates/desktop/heartbeat",
                    timeout=5,
                    json={"version": clean, "can_self_update": can_self_update},
                )
            except ServiceError:
                return
            return
        self._desktop_version = clean
        self._desktop_can_update = can_self_update
        self._desktop_seen_at = time.time()
        if self._pending_desktop and not is_newer_version(
            clean, self._pending_desktop.target_version
        ):
            self._pending_desktop = None

    async def ack_desktop_update(self, request_id: str) -> bool:
        if self._control_url():
            data = await self._control_request(
                "POST",
                "/updates/desktop/ack",
                timeout=10,
                json={"request_id": request_id},
            )
            return bool(data.get("acknowledged", False))
        if self._pending_desktop and self._pending_desktop.request_id == request_id:
            self._pending_desktop = None
            return True
        return False

    async def status(self) -> UpdateStatusResponse:
        if self._control_url():
            data = await self._control_request("GET", "/updates/status", timeout=10)
            return UpdateStatusResponse.model_validate(data)
        return UpdateStatusResponse(
            state="idle",
            current_version=getattr(self.services.settings, "app_version", "0.1.0"),
            desktop=self._fallback_desktop_status(),
            pending_desktop_update=self._pending_desktop,
        )

    async def check(self) -> UpdateCheckResult:
        if self._control_url():
            data = await self._control_request("GET", "/updates", timeout=30)
            return UpdateCheckResult.model_validate(data)

        current = getattr(self.services.settings, "app_version", "0.1.0")
        api_url = getattr(self.services.settings, "update_api_url", None)
        if not api_url:
            repo = getattr(self.services.settings, "update_repo", "ghalrym/Kadan")
            api_url = f"https://api.github.com/repos/{repo}/releases/latest"

        try:
            response = await self.services.client.get(
                api_url,
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": f"Kadan/{current}",
                },
                follow_redirects=True,
                timeout=httpx.Timeout(20, connect=5),
            )
        except httpx.RequestError as error:
            raise ServiceError(
                "Could not check for updates right now. Check your internet connection and try again.",
                502,
                "update_check_failed",
            ) from error

        if response.status_code == 404:
            return UpdateCheckResult(
                current_version=current,
                update_available=False,
                can_self_update=False,
                install_kind="manual",
                reason="No published releases were found yet.",
                desktop=self._fallback_desktop_status(),
            )
        if not response.is_success:
            raise ServiceError(
                f"Update check failed with HTTP {response.status_code}. Try again later.",
                502,
                "update_check_failed",
            )

        payload = response.json()
        tag = str(payload.get("tag_name") or "").strip()
        latest = tag.lstrip("vV") or None
        self._latest_known = latest
        raw_notes = payload.get("body")
        notes = None
        if isinstance(raw_notes, str) and raw_notes.strip():
            trimmed = raw_notes.strip()
            notes = trimmed if len(trimmed) <= 1500 else trimmed[:1500] + "…"
        update_available = bool(latest) and is_newer_version(current, latest)
        return UpdateCheckResult(
            current_version=current,
            latest_version=latest,
            update_available=update_available,
            release_tag=tag or None,
            release_url=payload.get("html_url") or None,
            release_notes=notes,
            published_at=payload.get("published_at") or None,
            can_self_update=False,
            install_kind="manual",
            reason="Automatic server updates require running the Kadan Server desktop app.",
            desktop=self._fallback_desktop_status(latest),
        )

    async def stage(self, target_version: str | None = None) -> StageUpdateResponse:
        if self._control_url():
            data = await self._control_request(
                "POST",
                "/updates/stage",
                timeout=300,
                json={"target_version": target_version},
            )
            return StageUpdateResponse.model_validate(data)

        check = await self.check()
        if not check.latest_version:
            raise ServiceError("No published Kadan releases were found.", 422)
        if (
            target_version
            and target_version.strip().lstrip("vV") != check.latest_version
        ):
            raise ServiceError(
                f"The latest available release is {check.latest_version}, not {target_version.strip().lstrip('vV')}.",
                422,
            )
        if (
            check.desktop.connected
            and check.desktop.update_available
            and check.desktop.can_self_update
        ):
            return StageUpdateResponse(
                ok=True,
                target_version=check.latest_version,
                server_staged=False,
                desktop_update_available=True,
                message=(
                    f"Ready to update Kadan Desktop to {check.latest_version}. "
                    "This server deployment itself must be updated manually."
                ),
            )
        raise ServiceError(
            check.reason
            or "Automatic updates are only available when running the Kadan Server desktop app.",
            422,
            "self_update_unsupported",
        )

    async def apply(
        self, target_version: str | None = None, include_desktop: bool = True
    ) -> ApplyUpdateResponse:
        if self._control_url():
            data = await self._control_request(
                "POST",
                "/updates/apply",
                timeout=30,
                json={
                    "target_version": target_version,
                    "include_desktop": include_desktop,
                },
            )
            return ApplyUpdateResponse.model_validate(data)

        target = target_version or self._latest_known
        desktop = self._fallback_desktop_status(target)
        if (
            include_desktop
            and target
            and desktop.connected
            and desktop.can_self_update
            and desktop.update_available
        ):
            pending = PendingDesktopUpdate(
                request_id=str(uuid4()),
                target_version=target,
                requested_at=int(time.time()),
            )
            self._pending_desktop = pending
            return ApplyUpdateResponse(
                ok=True,
                target_version=target,
                server_restarting=False,
                desktop_update_requested=True,
                pending_desktop_update=pending,
                message=f"Requested Kadan Desktop to update to {target} and restart.",
            )
        raise ServiceError(
            "Automatic server updates require running the Kadan Server desktop app.",
            422,
            "self_update_unsupported",
        )
