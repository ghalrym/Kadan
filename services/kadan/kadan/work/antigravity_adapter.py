"""Google Antigravity coding harness adapter powered by the shared ACP client (`antigravity-acp`)."""

from __future__ import annotations

import os
import shutil
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..api.config import ServiceError
from .acp_client import AcpHarnessSession
from .contract import (
    HarnessAdapter,
    HarnessAuthStatus,
    HarnessAvailability,
    HarnessError,
    HarnessModelInfo,
    HarnessSession,
    SessionConfig,
)
from .stdio_jsonrpc import ProcessFactory

if TYPE_CHECKING:
    from ..api.services import LocalServices

ANTIGRAVITY_MODELS: list[HarnessModelInfo] = [
    HarnessModelInfo(
        id="gemini-2.5-pro",
        label="Gemini 2.5 Pro",
        description="High-accuracy coding and multi-file reasoning.",
        options={"effort": ["low", "medium", "high"]},
    ),
    HarnessModelInfo(
        id="gemini-2.5-flash",
        label="Gemini 2.5 Flash",
        description="Fast turnaround for routine edits and questions.",
    ),
]


def find_antigravity_command() -> list[str] | None:
    override = os.environ.get("ANTIGRAVITY_ACP_BIN", "").strip()
    if override and Path(override).is_file():
        return [override]
    on_path = shutil.which("antigravity-acp")
    if on_path:
        return [on_path]
    # Discover managed binary installed alongside T3 Code or ~/.antigravity.
    t3_root = Path.home() / ".t3" / "userdata" / "providers" / "antigravity"
    if t3_root.is_dir():
        for candidate in sorted(t3_root.glob("*/antigravity-acp/bin/antigravity-acp")):
            if candidate.is_file():
                return [str(candidate)]
    local_bin = Path.home() / ".antigravity" / "bin" / "antigravity-acp"
    if local_bin.is_file():
        return [str(local_bin)]
    agy = shutil.which("agy")
    if agy:
        return [agy, "acp"]
    return None


def find_antigravity_binary() -> str | None:
    cmd = find_antigravity_command()
    return cmd[0] if cmd else None


def _has_local_antigravity_auth() -> bool:
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return True
    for candidate in (
        Path.home() / ".antigravity" / "credentials.json",
        Path.home() / ".config" / "antigravity" / "auth.json",
    ):
        if candidate.is_file():
            return True
    t3_root = Path.home() / ".t3" / "userdata" / "providers" / "antigravity"
    if t3_root.is_dir() and any(t3_root.iterdir()):
        return True
    return False


class AntigravityHarnessAdapter(HarnessAdapter):
    """Bundled Google Antigravity adapter over ACP stdio JSON-RPC (v2 with v1 fallback)."""

    harness_id = "antigravity"
    display_name = "Antigravity"
    description = "Google Antigravity coding harness over Agent Client Protocol."

    def __init__(self, *, process_factory: ProcessFactory | None = None):
        self._process_factory = process_factory

    async def _stored_credential(
        self, services: LocalServices
    ) -> tuple[str | None, str | None]:
        try:
            cfg = await services.json_request(
                "Tsuchi", "GET", "/api/work/harnesses/antigravity"
            )
        except ServiceError:
            cfg = None
        if isinstance(cfg, dict) and cfg.get("credential"):
            return (
                str(cfg["credential"]),
                cfg.get("account_label") or "Google Antigravity",
            )
        return None, None

    async def check_availability(self, services: LocalServices) -> HarnessAvailability:
        command = find_antigravity_command()
        binary = command[0] if command else None
        cred, label = await self._stored_credential(services)
        if not binary:
            auth = HarnessAuthStatus(
                status="unavailable",
                login_method="api_key",
                message="Google Antigravity is not installed on this computer yet.",
            )
        elif cred:
            auth = HarnessAuthStatus(
                status="ready",
                login_method="api_key",
                account_label=label,
                message=f"Connected ({label}).",
            )
        elif _has_local_antigravity_auth():
            auth = HarnessAuthStatus(
                status="ready",
                login_method="browser",
                account_label="Local Antigravity sign-in",
                message="Connected using this computer's Antigravity sign-in.",
            )
        else:
            auth = HarnessAuthStatus(
                status="sign_in_required",
                login_method="api_key",
                authorize_url="https://aistudio.google.com/apikey",
                needs_code=True,
                message="Sign in to Google Antigravity or enter your Google AI key.",
            )
        return HarnessAvailability(
            harness_id=self.harness_id,
            display_name=self.display_name,
            description=self.description,
            installed=binary is not None,
            binary_path=binary,
            auth=auth,
            models=ANTIGRAVITY_MODELS,
            default_model=ANTIGRAVITY_MODELS[0].id,
            supports_steer=False,
            supports_resume=True,
            supports_approvals=True,
            supports_questions=True,
        )

    async def start_sign_in(
        self, services: LocalServices, payload: dict[str, Any] | None = None
    ) -> HarnessAuthStatus:
        return HarnessAuthStatus(
            status="waiting",
            login_method="api_key",
            authorize_url="https://aistudio.google.com/apikey",
            needs_code=True,
            message="Open Google AI Studio to get a key, then paste it below.",
        )

    async def submit_sign_in(
        self, services: LocalServices, payload: dict[str, Any]
    ) -> HarnessAuthStatus:
        credential = str(payload.get("credential") or payload.get("code") or "").strip()
        if not credential:
            raise HarnessError("Enter your Google Antigravity credential to connect.")
        label = str(payload.get("account_label") or "Google Antigravity").strip()
        await services.json_request(
            "Tsuchi",
            "PUT",
            "/api/work/harnesses/antigravity",
            json={
                "credential": credential,
                "auth_mode": "api_key",
                "account_label": label,
                "metadata": {},
            },
        )
        return HarnessAuthStatus(
            status="ready",
            login_method="api_key",
            account_label=label,
            message=f"Connected ({label}).",
        )

    async def sign_out(self, services: LocalServices) -> HarnessAuthStatus:
        with suppress(ServiceError):
            await services.json_request(
                "Tsuchi", "DELETE", "/api/work/harnesses/antigravity"
            )
        return HarnessAuthStatus(
            status="sign_in_required",
            login_method="api_key",
            needs_code=True,
            message="Signed out of Antigravity.",
        )

    async def open_session(
        self, config: SessionConfig, services: LocalServices
    ) -> HarnessSession:
        command = find_antigravity_command()
        if self._process_factory is None and not command:
            raise HarnessError("Google Antigravity is not installed on this computer.")
        cred, _ = await self._stored_credential(services)
        env_extra: dict[str, str] = {}
        if cred:
            env_extra["GEMINI_API_KEY"] = cred
            env_extra["GOOGLE_API_KEY"] = cred
        return AcpHarnessSession(
            config,
            command=command or ["antigravity-acp"],
            display_name=self.display_name,
            env_extra=env_extra,
            process_factory=self._process_factory,
        )
