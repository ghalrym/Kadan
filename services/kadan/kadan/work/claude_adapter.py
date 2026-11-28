"""Claude Code coding harness adapter (`claude -p --input-format stream-json --output-format stream-json`).

Runs `@anthropic-ai/claude-code` in full coding-harness mode bound to the thread's
local project folder, supporting live streamed text/tool progress,
interactive approvals & questions via `control_request` / `control_response`, interrupt,
and `--resume` session continuity.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..api.config import ServiceError
from .auth_flows import cli_logins
from .contract import (
    HarnessAdapter,
    HarnessAuthStatus,
    HarnessAvailability,
    HarnessError,
    HarnessModelInfo,
    HarnessSession,
    ModelCatalogCache,
    NormalizedEvent,
    SessionConfig,
    fingerprint_secret,
)
from .stdio_jsonrpc import ProcessFactory, StdioProcess

if TYPE_CHECKING:
    from ..api.services import LocalServices

logger = logging.getLogger(__name__)


def _parse_claude_models(
    entries: list[dict[str, Any]],
) -> tuple[list[HarnessModelInfo], str | None]:
    """Normalize Claude Code `initialize` `models` entries into `HarnessModelInfo`."""
    valid_entries = [e for e in entries if isinstance(e, dict)]
    default_resolved: str | None = None
    non_default_resolved: set[str] = set()

    for entry in valid_entries:
        val = str(entry.get("value") or "").strip()
        resolved = str(entry.get("resolvedModel") or "").strip()
        if val == "default" and resolved:
            default_resolved = resolved
        elif val != "default" and resolved:
            non_default_resolved.add(resolved)

    models: list[HarnessModelInfo] = []
    seen: set[str] = set()

    for entry in valid_entries:
        val = str(entry.get("value") or "").strip()
        resolved = str(entry.get("resolvedModel") or "").strip()
        # Skip the synthetic "default" alias when a concrete entry for the same model exists.
        if val == "default" and resolved and resolved in non_default_resolved:
            continue
        model_id = resolved or val
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)

        label = str(entry.get("displayName") or model_id).strip() or model_id
        description = str(entry.get("description") or "").strip()

        efforts: list[str] = []
        raw_efforts = entry.get("supportedEffortLevels")
        if isinstance(raw_efforts, list):
            for level in raw_efforts:
                if (
                    isinstance(level, str)
                    and level.strip()
                    and level.strip() not in efforts
                ):
                    efforts.append(level.strip())

        options = {"effort": efforts} if efforts else {}
        models.append(
            HarnessModelInfo(
                id=model_id,
                label=label,
                description=description,
                options=options,
            )
        )

    default_model: str | None = None
    if default_resolved and default_resolved in seen:
        default_model = default_resolved
    elif models:
        default_model = models[0].id
    return models, default_model


def _find_claude_binary() -> str | None:
    override = os.environ.get("KADAN_CLAUDE_BIN", "").strip()
    if override and Path(override).is_file():
        return override
    return shutil.which("claude")


def _local_claude_signed_in() -> bool:
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY"):
        return True
    cred_path = Path.home() / ".claude" / ".credentials.json"
    return cred_path.is_file()


def _claude_permission_mode(mode: str) -> str:
    if mode == "full_auto":
        return "bypassPermissions"
    if mode == "auto_edit":
        return "acceptEdits"
    return "default"


class ClaudeHarnessSession(HarnessSession):
    """Live Claude Code `stream-json` coding session."""

    def __init__(
        self,
        config: SessionConfig,
        *,
        binary: str,
        credential: str | None = None,
        process_factory: ProcessFactory | None = None,
    ):
        self.config = config
        self.binary = binary
        self._credential = credential
        self._process_factory = process_factory
        self._transport: StdioProcess | None = None
        self._session_id: str | None = config.harness_session_id
        self._pending_controls: dict[str, dict[str, Any]] = {}
        self._interrupted = False

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def _build_command(self) -> tuple[list[str], dict[str, str]]:
        env = dict(os.environ)
        env.update(self.config.env_overrides)
        if self._credential:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = self._credential
            env.pop("ANTHROPIC_API_KEY", None)

        cmd = [
            self.binary,
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--permission-prompt-tool",
            "stdio",
            "--permission-mode",
            _claude_permission_mode(self.config.permission_mode),
        ]
        if self.config.model:
            cmd.extend(["--model", self.config.model])
        effort = self.config.model_options.get("effort")
        if isinstance(effort, str) and effort:
            cmd.extend(["--effort", effort])
        if self._session_id:
            cmd.extend(["--resume", self._session_id])
        return cmd, env

    async def _ensure_started(self) -> StdioProcess:
        if self._transport is not None:
            return self._transport
        cmd, env = self._build_command()
        if self._process_factory is not None:
            self._transport = await self._process_factory(
                cmd, self.config.working_dir, env
            )
        else:
            self._transport = await StdioProcess.spawn(
                cmd,
                cwd=self.config.working_dir,
                env=env,
                not_found_message="Claude Code is not installed on this server.",
            )
        return self._transport

    async def start_turn(self, prompt: str) -> AsyncIterator[NormalizedEvent]:
        self._interrupted = False
        transport = await self._ensure_started()
        await transport.write_json(
            {
                "type": "user",
                "message": {"role": "user", "content": prompt},
            }
        )

        final_text = ""
        announced_session = False

        while True:
            event = await transport.read_json()
            if event is None:
                if self._interrupted:
                    yield NormalizedEvent(
                        kind="turn_completed",
                        text=final_text,
                        terminal_status="interrupted",
                    )
                else:
                    detail = await transport.error_summary()
                    yield NormalizedEvent(
                        kind="turn_completed",
                        text=final_text,
                        terminal_status="failed",
                        error=detail,
                    )
                return

            sid = event.get("session_id") or event.get("sessionId")
            if isinstance(sid, str) and sid and sid != self._session_id:
                self._session_id = sid
                announced_session = True
                yield NormalizedEvent(kind="session_started", session_id=sid)
            elif not announced_session and self._session_id:
                announced_session = True
                yield NormalizedEvent(
                    kind="session_started", session_id=self._session_id
                )

            ev_type = str(event.get("type") or "")

            # Interactive permission approval / user question control requests.
            if ev_type == "control_request":
                req_id = str(event.get("request_id") or "")
                request = (
                    event.get("request")
                    if isinstance(event.get("request"), dict)
                    else {}
                )
                subtype = str(request.get("subtype") or "")
                if subtype == "can_use_tool":
                    tool_name = str(request.get("tool_name") or "tool")
                    tool_input = (
                        request.get("input")
                        if isinstance(request.get("input"), dict)
                        else {}
                    )
                    self._pending_controls[req_id] = {
                        "tool_name": tool_name,
                        "input": tool_input,
                    }
                    if tool_name == "AskUserQuestion":
                        questions = tool_input.get("questions") or []
                        first_q = (
                            questions[0]
                            if isinstance(questions, list) and questions
                            else {}
                        )
                        yield NormalizedEvent(
                            kind="pending_question",
                            request_id=req_id,
                            title=str(
                                first_q.get("header") or "Question from Claude Code"
                            ),
                            text=str(
                                first_q.get("question")
                                or "Claude Code has a question before continuing."
                            ),
                            options=first_q.get("options")
                            if isinstance(first_q.get("options"), list)
                            else [],
                            tool_name=tool_name,
                            arguments=tool_input,
                            raw=request,
                        )
                    else:
                        prompt_text = str(
                            tool_input.get("command")
                            or tool_input.get("file_path")
                            or tool_input.get("description")
                            or f"Allow Claude Code to use {tool_name}?"
                        )
                        yield NormalizedEvent(
                            kind="pending_approval",
                            request_id=req_id,
                            title=f"Allow {tool_name}?",
                            text=prompt_text,
                            tool_name=tool_name,
                            arguments=tool_input,
                            options=[
                                {"label": "Approve", "value": "allow"},
                                {"label": "Deny", "value": "deny"},
                            ],
                            raw=request,
                        )
                continue

            if ev_type == "stream_event":
                inner = (
                    event.get("event") if isinstance(event.get("event"), dict) else {}
                )
                inner_type = str(inner.get("type") or "")
                delta = (
                    inner.get("delta") if isinstance(inner.get("delta"), dict) else {}
                )
                if inner_type == "content_block_delta":
                    delta_type = str(delta.get("type") or "")
                    if delta_type == "text_delta":
                        chunk = str(delta.get("text") or "")
                        if chunk:
                            final_text += chunk
                            yield NormalizedEvent(kind="text_delta", text=chunk)
                    elif delta_type == "thinking_delta":
                        chunk = str(delta.get("thinking") or "")
                        if chunk:
                            yield NormalizedEvent(kind="reasoning_delta", text=chunk)
                continue

            if ev_type == "assistant":
                msg = (
                    event.get("message")
                    if isinstance(event.get("message"), dict)
                    else {}
                )
                content = msg.get("content")
                if isinstance(content, list):
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        ptype = part.get("type")
                        if ptype == "text" and isinstance(part.get("text"), str):
                            final_text = part["text"]
                            yield NormalizedEvent(
                                kind="assistant_message", text=part["text"]
                            )
                        elif ptype == "tool_use":
                            name = str(part.get("name") or "tool")
                            inp = (
                                part.get("input")
                                if isinstance(part.get("input"), dict)
                                else {}
                            )
                            yield NormalizedEvent(
                                kind="tool_call",
                                call_id=str(part.get("id") or ""),
                                tool_name=name,
                                arguments=inp,
                                raw=part,
                            )
                            if name in {"Edit", "Write", "MultiEdit", "NotebookEdit"}:
                                fp = inp.get("file_path") or inp.get("path")
                                if isinstance(fp, str) and fp:
                                    yield NormalizedEvent(
                                        kind="file_change",
                                        file_path=fp,
                                        change_type="write"
                                        if name == "Write"
                                        else "edit",
                                    )
                continue

            if ev_type == "user":
                msg = (
                    event.get("message")
                    if isinstance(event.get("message"), dict)
                    else {}
                )
                content = msg.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "tool_result":
                            raw_out = part.get("content", "")
                            if not isinstance(raw_out, str):
                                raw_out = json.dumps(raw_out, ensure_ascii=False)
                            yield NormalizedEvent(
                                kind="tool_result",
                                call_id=str(part.get("tool_use_id") or ""),
                                text=raw_out[:8000],
                                is_error=bool(part.get("is_error")),
                                raw=part,
                            )
                continue

            if ev_type == "result":
                subtype = str(event.get("subtype") or "success").lower()
                is_err = bool(event.get("is_error")) or subtype not in {"success"}
                res_text = str(event.get("result") or "")
                if res_text and not final_text:
                    final_text = res_text
                if self._interrupted or subtype in {"interrupted", "cancelled"}:
                    terminal = "interrupted"
                    err_text = None
                elif is_err:
                    terminal = "failed"
                    err_text = res_text or str(
                        event.get("errors") or "Claude Code turn failed."
                    )
                else:
                    terminal = "completed"
                    err_text = None
                yield NormalizedEvent(
                    kind="turn_completed",
                    text=final_text,
                    terminal_status=terminal,
                    error=err_text,
                    raw=event,
                )
                return

    async def interrupt(self) -> None:
        self._interrupted = True
        if self._transport is None:
            return
        with suppress(Exception):
            await self._transport.write_json(
                {
                    "type": "control_request",
                    "request_id": "kadan-interrupt",
                    "request": {"subtype": "interrupt"},
                }
            )

    async def respond_pending(self, request_id: str, response: dict[str, Any]) -> None:
        if self._transport is None:
            raise HarnessError("Claude Code session is not running.")
        meta = self._pending_controls.pop(request_id, None)
        if meta is None:
            raise HarnessError("That pending Claude Code request is no longer active.")
        tool_name = meta["tool_name"]
        original_input = meta["input"]

        if tool_name == "AskUserQuestion":
            updated_input = dict(original_input)
            answers = response.get("answers")
            if answers is None:
                answer = str(response.get("answer") or response.get("text") or "")
                updated_input["answers"] = {"answer": answer}
            else:
                updated_input["answers"] = answers
            payload = {"behavior": "allow", "updatedInput": updated_input}
        else:
            decision = str(response.get("decision") or "deny").lower()
            if decision in {"approve", "allow", "accept", "approve_session"}:
                payload = {"behavior": "allow", "updatedInput": original_input}
            else:
                payload = {
                    "behavior": "deny",
                    "message": str(
                        response.get("reason") or "The user declined this action."
                    ),
                }
        await self._transport.write_json(
            {
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": request_id,
                    "response": payload,
                },
            }
        )

    async def close(self) -> None:
        if self._transport is not None:
            await self._transport.close()
            self._transport = None


class ClaudeHarnessAdapter(HarnessAdapter):
    """Bundled Claude Code adapter using `claude -p --output-format stream-json`."""

    harness_id = "claude"
    display_name = "Claude Code"
    description = "Anthropic Claude"

    def __init__(self, *, process_factory: ProcessFactory | None = None):
        self._process_factory = process_factory
        self._model_cache = ModelCatalogCache()

    async def _stored_credential(
        self, services: LocalServices
    ) -> tuple[str | None, str | None]:
        try:
            cfg = await services.json_request(
                "Tsuchi", "GET", "/api/work/harnesses/claude"
            )
        except ServiceError:
            cfg = None
        if isinstance(cfg, dict) and cfg.get("credential"):
            return (
                str(cfg["credential"]),
                cfg.get("account_label") or "Claude subscription",
            )
        return None, None

    async def _discover_models(
        self, binary: str, credential: str | None
    ) -> tuple[list[HarnessModelInfo], str | None]:
        env = dict(os.environ)
        if credential:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = credential
            env.pop("ANTHROPIC_API_KEY", None)
        cmd = [
            binary,
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-prompt-tool",
            "stdio",
        ]
        transport: StdioProcess | None = None
        try:
            if self._process_factory is not None:
                transport = await self._process_factory(cmd, os.getcwd(), env)
            else:
                transport = await StdioProcess.spawn(
                    cmd,
                    cwd=os.getcwd(),
                    env=env,
                    not_found_message="Claude Code is not installed on this computer.",
                )
            req_id = "kadan-init-models"
            await transport.write_json(
                {
                    "type": "control_request",
                    "request_id": req_id,
                    "request": {"subtype": "initialize"},
                }
            )

            async def _read_init_models() -> tuple[list[HarnessModelInfo], str | None]:
                for _ in range(30):
                    frame = await transport.read_json()
                    if frame is None:
                        return [], None
                    if frame.get("type") != "control_response":
                        continue
                    outer = frame.get("response")
                    if not isinstance(outer, dict):
                        continue
                    if outer.get("request_id") not in {req_id, None}:
                        continue
                    inner = outer.get("response")
                    raw_models = (
                        inner.get("models") if isinstance(inner, dict) else None
                    )
                    if isinstance(raw_models, list):
                        return _parse_claude_models(raw_models)
                    return [], None
                return [], None

            return await asyncio.wait_for(_read_init_models(), timeout=15.0)
        except (HarnessError, TimeoutError, OSError) as exc:
            logger.warning("Claude Code model discovery failed: %s", exc)
            return [], None
        finally:
            if transport is not None:
                with suppress(Exception):
                    await transport.close()

    async def check_availability(self, services: LocalServices) -> HarnessAvailability:
        binary = _find_claude_binary()
        in_flight = cli_logins.status("claude")
        cred, label = await self._stored_credential(services)
        if not binary:
            auth = HarnessAuthStatus(
                status="unavailable",
                login_method="oauth_code",
                needs_code=True,
                message="Not installed on this computer.",
            )
        elif in_flight is not None and in_flight.status in {
            "starting",
            "waiting",
            "verifying",
        }:
            auth = in_flight
        elif cred:
            auth = HarnessAuthStatus(
                status="ready",
                login_method="oauth_code",
                account_label=label,
                message="Connected.",
            )
        elif _local_claude_signed_in():
            auth = HarnessAuthStatus(
                status="ready",
                login_method="oauth_code",
                account_label="Local sign-in",
                message="Connected.",
            )
        else:
            auth = HarnessAuthStatus(
                status="sign_in_required",
                login_method="oauth_code",
                needs_code=True,
                message="Sign in to connect.",
            )

        models: list[HarnessModelInfo] = []
        default_model: str | None = None
        if binary and auth.status == "ready":
            cache_key = f"{binary}:{fingerprint_secret(cred)}"
            models, default_model = await self._model_cache.get_or_load(
                cache_key,
                lambda: self._discover_models(binary, cred),
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
        binary = _find_claude_binary()
        if not binary:
            raise HarnessError("Claude Code is not installed on this server.")
        self._model_cache.invalidate()
        return await cli_logins.start("claude", binary, services)

    async def sign_in_status(self, services: LocalServices) -> HarnessAuthStatus:
        in_flight = cli_logins.status("claude")
        if in_flight is not None:
            return in_flight
        return (await self.check_availability(services)).auth

    async def submit_sign_in(
        self, services: LocalServices, payload: dict[str, Any]
    ) -> HarnessAuthStatus:
        code = str(payload.get("code") or "").strip()
        if not code:
            raise HarnessError("Paste the sign-in code from Claude's page.")
        self._model_cache.invalidate()
        return await cli_logins.submit_code("claude", code)

    async def sign_out(self, services: LocalServices) -> HarnessAuthStatus:
        await cli_logins.cancel("claude")
        self._model_cache.invalidate()
        with suppress(ServiceError):
            await services.json_request(
                "Tsuchi", "DELETE", "/api/work/harnesses/claude"
            )
        return HarnessAuthStatus(
            status="sign_in_required",
            login_method="oauth_code",
            needs_code=True,
            message="Signed out of Claude Code.",
        )

    async def open_session(
        self, config: SessionConfig, services: LocalServices
    ) -> HarnessSession:
        binary = _find_claude_binary() or "claude"
        if self._process_factory is None and not _find_claude_binary():
            raise HarnessError("Claude Code is not installed on this server.")
        cred, _ = await self._stored_credential(services)
        return ClaudeHarnessSession(
            config,
            binary=binary,
            credential=cred,
            process_factory=self._process_factory,
        )
