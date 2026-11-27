"""xAI Grok (Grok Build CLI) coding harness adapter powered by the shared ACP client."""

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

GROK_MODELS: list[HarnessModelInfo] = [
    HarnessModelInfo(
        id="grok-code-fast-1",
        label="Grok Code Fast 1",
        description="Fast agentic coding model tuned for software engineering.",
    ),
    HarnessModelInfo(
        id="grok-4",
        label="Grok 4",
        description="Flagship reasoning model for complex debugging and design.",
    ),
    HarnessModelInfo(
        id="grok-4-fast",
        label="Grok 4 Fast",
        description="Low-latency model for quick iterations.",
    ),
]


def find_grok_command() -> list[str] | None:
    override = os.environ.get("KADAN_GROK_BIN", "").strip()
    if override and Path(override).is_file():
        return [override, "--acp"]
    binary = shutil.which("grok")
    if binary:
        return [binary, "--acp"]
    npx = shutil.which("npx")
    if npx and os.environ.get("KADAN_GROK_ALLOW_NPX", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return [npx, "-y", "@xai-official/grok", "--acp"]
    return None


def _has_local_grok_auth() -> bool:
    if os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY"):
        return True
    for candidate in (
        Path.home() / ".grok" / "auth.json",
        Path.home() / ".config" / "grok" / "credentials.json",
    ):
        if candidate.is_file():
            return True
    return False


class GrokHarnessAdapter(HarnessAdapter):
    """Bundled xAI Grok adapter over ACP stdio JSON-RPC (v2 with v1 fallback)."""

    harness_id = "grok"
    display_name = "Grok"
    description = "xAI Grok"

    def __init__(self, *, process_factory: ProcessFactory | None = None):
        self._process_factory = process_factory

    async def _stored_credential(
        self, services: LocalServices
    ) -> tuple[str | None, str | None]:
        try:
            cfg = await services.json_request(
                "Tsuchi", "GET", "/api/work/harnesses/grok"
            )
        except ServiceError:
            cfg = None
        if isinstance(cfg, dict) and cfg.get("credential"):
            return str(cfg["credential"]), cfg.get("account_label") or "xAI Grok"
        return None, None

    async def check_availability(self, services: LocalServices) -> HarnessAvailability:
        cmd = find_grok_command()
        cred, label = await self._stored_credential(services)
        if not cmd:
            auth = HarnessAuthStatus(
                status="unavailable",
                login_method="api_key",
                authorize_url="https://console.x.ai/",
                message="Not installed on this computer.",
            )
        elif cred:
            auth = HarnessAuthStatus(
                status="ready",
                login_method="api_key",
                account_label=label,
                message="Connected.",
            )
        elif _has_local_grok_auth():
            auth = HarnessAuthStatus(
                status="ready",
                login_method="api_key",
                account_label="Local key",
                message="Connected.",
            )
        else:
            auth = HarnessAuthStatus(
                status="sign_in_required",
                login_method="api_key",
                authorize_url="https://console.x.ai/",
                needs_code=True,
                message="Paste your xAI access key.",
            )
        return HarnessAvailability(
            harness_id=self.harness_id,
            display_name=self.display_name,
            description=self.description,
            installed=cmd is not None,
            binary_path=cmd[0] if cmd else None,
            auth=auth,
            models=GROK_MODELS,
            default_model=GROK_MODELS[0].id,
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
            authorize_url="https://console.x.ai/",
            needs_code=True,
            message="Open xAI Console to create an access key, then paste it below.",
        )

    async def submit_sign_in(
        self, services: LocalServices, payload: dict[str, Any]
    ) -> HarnessAuthStatus:
        credential = str(payload.get("credential") or payload.get("code") or "").strip()
        if not credential:
            raise HarnessError("Enter your xAI access key to connect Grok.")
        label = str(payload.get("account_label") or "xAI Grok").strip()
        await services.json_request(
            "Tsuchi",
            "PUT",
            "/api/work/harnesses/grok",
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
            await services.json_request("Tsuchi", "DELETE", "/api/work/harnesses/grok")
        return HarnessAuthStatus(
            status="sign_in_required",
            login_method="api_key",
            authorize_url="https://console.x.ai/",
            needs_code=True,
            message="Signed out of Grok.",
        )

    async def open_session(
        self, config: SessionConfig, services: LocalServices
    ) -> HarnessSession:
        cmd = find_grok_command() or ["grok", "--acp"]
        if self._process_factory is None and not find_grok_command():
            raise HarnessError("Grok CLI is not installed on this server.")
        cred, _ = await self._stored_credential(services)
        env_extra: dict[str, str] = {}
        if cred:
            env_extra["XAI_API_KEY"] = cred
            env_extra["GROK_API_KEY"] = cred
        return AcpHarnessSession(
            config,
            command=cmd,
            display_name=self.display_name,
            env_extra=env_extra,
            process_factory=self._process_factory,
        )
