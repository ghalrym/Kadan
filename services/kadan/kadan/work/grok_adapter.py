"""xAI Grok (Grok Build CLI) coding harness adapter powered by the shared ACP client."""

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
_GROK_MODEL_LINE_RE = re.compile(r"^\s*([*-])\s+(\S+)(?:\s+(.*))?$")


def _format_grok_label(model_id: str) -> str:
    parts = [part for part in model_id.split("-") if part]
    if not parts:
        return model_id
    return " ".join(part if part[0].isdigit() else part.capitalize() for part in parts)


def _parse_grok_models_output(
    text: str,
) -> tuple[list[HarnessModelInfo], str | None]:
    """Parse `grok models` stdout (JSON or text bullet list) into `HarnessModelInfo`."""
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
                    mid = item.strip()
                    seen.add(mid)
                    models.append(
                        HarnessModelInfo(id=mid, label=_format_grok_label(mid))
                    )
                elif isinstance(item, dict):
                    mid = str(
                        item.get("id") or item.get("modelId") or item.get("model") or ""
                    ).strip()
                    if not mid or mid in seen:
                        continue
                    seen.add(mid)
                    label = str(
                        item.get("displayName")
                        or item.get("name")
                        or _format_grok_label(mid)
                    ).strip()
                    desc = str(item.get("description") or "").strip()
                    models.append(
                        HarnessModelInfo(
                            id=mid,
                            label=label or _format_grok_label(mid),
                            description=desc,
                        )
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
        match = _GROK_MODEL_LINE_RE.match(raw_line)
        if not match:
            continue
        bullet, mid, rest = (
            match.group(1),
            match.group(2).strip(),
            (match.group(3) or "").strip(),
        )
        is_def = bullet == "*" or "(default)" in rest.lower()
        rest_clean = re.sub(r"\(default\)", "", rest, flags=re.IGNORECASE).strip(" -—:")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        models.append(
            HarnessModelInfo(
                id=mid,
                label=_format_grok_label(mid),
                description=rest_clean,
            )
        )
        if is_def and default_model is None:
            default_model = mid

    if default_model and default_model not in seen:
        default_model = models[0].id if models else None
    elif not default_model and models:
        default_model = models[0].id
    return models, default_model


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
        self._model_cache = ModelCatalogCache()

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

    async def _discover_models(
        self, cmd: list[str], credential: str | None
    ) -> tuple[list[HarnessModelInfo], str | None]:
        env_extra: dict[str, str] = {}
        if credential:
            env_extra["XAI_API_KEY"] = credential
            env_extra["GROK_API_KEY"] = credential

        if self._process_factory is not None:
            return await discover_acp_models(
                cmd,
                display_name=self.display_name,
                env_extra=env_extra,
                process_factory=self._process_factory,
            )

        # Prefer `grok models` subcommand when driving the native Grok CLI.
        if cmd and Path(cmd[0]).name != "npx":
            env = dict(os.environ)
            env.update(env_extra)
            proc: asyncio.subprocess.Process | None = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    cmd[0],
                    "models",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
                if proc.returncode == 0:
                    models, default_model = _parse_grok_models_output(
                        stdout.decode("utf-8", errors="replace")
                    )
                    if models:
                        return models, default_model
            except (TimeoutError, OSError) as exc:
                logger.warning("Grok CLI model discovery failed: %s", exc)
            finally:
                if proc is not None and proc.returncode is None:
                    with suppress(ProcessLookupError):
                        proc.kill()

        return await discover_acp_models(
            cmd,
            display_name=self.display_name,
            env_extra=env_extra,
        )

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

        models: list[HarnessModelInfo] = []
        default_model: str | None = None
        if cmd and auth.status == "ready":
            cache_key = f"{' '.join(cmd)}:{fingerprint_secret(cred)}"
            models, default_model = await self._model_cache.get_or_load(
                cache_key,
                lambda: self._discover_models(cmd, cred),
            )

        return HarnessAvailability(
            harness_id=self.harness_id,
            display_name=self.display_name,
            description=self.description,
            installed=cmd is not None,
            binary_path=cmd[0] if cmd else None,
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
