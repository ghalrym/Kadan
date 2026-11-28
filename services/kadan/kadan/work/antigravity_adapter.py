"""Google Antigravity coding harness adapter powered by the shared ACP client (`antigravity-acp`)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..api.config import ServiceError
from .acp_client import AcpHarnessSession, discover_acp_models
from .contract import (
    HarnessAdapter,
    HarnessAuthStatus,
    HarnessAvailability,
    HarnessError,
    HarnessModelInfo,
    HarnessSession,
    ModelCatalogCache,
    SessionConfig,
    fingerprint_secret,
)
from .stdio_jsonrpc import ProcessFactory

if TYPE_CHECKING:
    from ..api.services import LocalServices

logger = logging.getLogger(__name__)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
_MODEL_BULLET_RE = re.compile(r"^\s*[-*•]\s+(\S+)(?:\s+(.*))?$")


def _parse_agy_models_output(
    text: str,
) -> tuple[list[HarnessModelInfo], str | None]:
    """Parse `agy models` stdout (JSON or plain text list) into `HarnessModelInfo`."""
    cleaned = _ANSI_RE.sub("", text).strip()
    if not cleaned:
        return [], None

    if cleaned.startswith(("{", "[")):
        with suppress(json.JSONDecodeError):
            parsed = json.loads(cleaned)
            raw_list = (
                parsed
                if isinstance(parsed, list)
                else (
                    parsed.get("models")
                    or parsed.get("availableModels")
                    or parsed.get("data")
                    or []
                )
                if isinstance(parsed, dict)
                else []
            )
            models: list[HarnessModelInfo] = []
            seen: set[str] = set()
            default_model: str | None = (
                str(parsed.get("defaultModel") or parsed.get("default") or "").strip()
                or None
                if isinstance(parsed, dict)
                else None
            )
            for item in raw_list:
                if isinstance(item, str) and item.strip() and item.strip() not in seen:
                    seen.add(item.strip())
                    models.append(HarnessModelInfo(id=item.strip(), label=item.strip()))
                elif isinstance(item, dict):
                    mid = str(
                        item.get("id") or item.get("modelId") or item.get("model") or ""
                    ).strip()
                    if not mid or mid in seen:
                        continue
                    seen.add(mid)
                    label = str(
                        item.get("displayName") or item.get("name") or mid
                    ).strip()
                    desc = str(item.get("description") or "").strip()
                    models.append(
                        HarnessModelInfo(id=mid, label=label or mid, description=desc)
                    )
                    if item.get("isDefault") is True and default_model is None:
                        default_model = mid
            if models:
                return models, (
                    default_model if default_model in seen else models[0].id
                )

    models = []
    seen = set()
    default_model = None
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("default model:"):
            cand = line.split(":", 1)[1].strip()
            if cand:
                default_model = cand
            continue
        match = _MODEL_BULLET_RE.match(raw_line)
        if not match:
            continue
        mid = match.group(1).strip()
        rest = (match.group(2) or "").strip()
        is_def = "(default)" in rest.lower() or raw_line.lstrip().startswith("*")
        rest_clean = re.sub(r"\(default\)", "", rest, flags=re.IGNORECASE).strip(" -—:")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        models.append(
            HarnessModelInfo(
                id=mid,
                label=rest_clean or mid,
            )
        )
        if is_def and default_model is None:
            default_model = mid

    if default_model and default_model not in seen:
        default_model = models[0].id if models else None
    elif not default_model and models:
        default_model = models[0].id
    return models, default_model


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


def _has_local_antigravity_auth(binary: str | None = None) -> bool:
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return True
    for candidate in (
        Path.home() / ".antigravity" / "credentials.json",
        Path.home() / ".config" / "antigravity" / "auth.json",
    ):
        if candidate.is_file():
            return True
    t3_root = Path.home() / ".t3" / "userdata" / "providers" / "antigravity"
    if binary and t3_root.is_dir():
        with suppress(OSError, ValueError):
            if Path(binary).resolve().is_relative_to(t3_root.resolve()):
                return any(t3_root.iterdir())
    return False


class AntigravityHarnessAdapter(HarnessAdapter):
    """Bundled Google Antigravity adapter over ACP stdio JSON-RPC (v2 with v1 fallback)."""

    harness_id = "antigravity"
    display_name = "Antigravity"
    description = "Google Antigravity"

    def __init__(self, *, process_factory: ProcessFactory | None = None):
        self._process_factory = process_factory
        self._model_cache = ModelCatalogCache()

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

    async def _discover_models(
        self, command: list[str], credential: str | None
    ) -> tuple[list[HarnessModelInfo], str | None]:
        env_extra: dict[str, str] = {}
        if credential:
            env_extra["GEMINI_API_KEY"] = credential
            env_extra["GOOGLE_API_KEY"] = credential

        models, default_model = await discover_acp_models(
            command,
            display_name=self.display_name,
            env_extra=env_extra,
            process_factory=self._process_factory,
        )
        if models or self._process_factory is not None:
            return models, default_model

        # Fallback: `agy models` subcommand when driving the `agy` CLI.
        if Path(command[0]).name == "agy":
            env = dict(os.environ)
            env.update(env_extra)
            proc: asyncio.subprocess.Process | None = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    command[0],
                    "models",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
                if proc.returncode == 0:
                    return _parse_agy_models_output(
                        stdout.decode("utf-8", errors="replace")
                    )
            except (TimeoutError, OSError) as exc:
                logger.warning("Antigravity CLI model discovery failed: %s", exc)
            finally:
                if proc is not None and proc.returncode is None:
                    with suppress(ProcessLookupError):
                        proc.kill()
        return [], None

    async def check_availability(self, services: LocalServices) -> HarnessAvailability:
        command = find_antigravity_command()
        binary = command[0] if command else None
        cred, label = await self._stored_credential(services)
        if not binary:
            auth = HarnessAuthStatus(
                status="unavailable",
                login_method="api_key",
                message="Not installed on this computer.",
            )
        elif cred:
            auth = HarnessAuthStatus(
                status="ready",
                login_method="api_key",
                account_label=label,
                message="Connected.",
            )
        elif _has_local_antigravity_auth(binary):
            auth = HarnessAuthStatus(
                status="ready",
                login_method="browser",
                account_label="Local sign-in",
                message="Connected.",
            )
        else:
            auth = HarnessAuthStatus(
                status="sign_in_required",
                login_method="api_key",
                authorize_url="https://aistudio.google.com/apikey",
                needs_code=True,
                message="Paste your Google AI key.",
            )

        models: list[HarnessModelInfo] = []
        default_model: str | None = None
        if command and auth.status == "ready":
            cache_key = f"{' '.join(command)}:{fingerprint_secret(cred)}"
            models, default_model = await self._model_cache.get_or_load(
                cache_key,
                lambda: self._discover_models(command, cred),
            )

        return HarnessAvailability(
            harness_id=self.harness_id,
            display_name=self.display_name,
            description=self.description,
            installed=binary is not None,
            binary_path=binary,
            auth=auth,
            models=models,
            default_model=default_model,
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
        self._model_cache.invalidate()
        return HarnessAuthStatus(
            status="ready",
            login_method="api_key",
            account_label=label,
            message=f"Connected ({label}).",
        )

    async def sign_out(self, services: LocalServices) -> HarnessAuthStatus:
        self._model_cache.invalidate()
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
