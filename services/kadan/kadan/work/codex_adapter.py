"""ChatGPT Codex coding harness adapter (`codex app-server` over stdio JSON-RPC).

Drives `@openai/codex` in full coding-harness mode (workspace tools, shell, file edits,
approvals, user questions, live turn steering, interrupt, and thread resume) bound to the
thread's project root or isolated Git worktree.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
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
    NormalizedEvent,
    SessionConfig,
)
from .stdio_jsonrpc import JsonRpcStdioClient, ProcessFactory, StdioProcess

if TYPE_CHECKING:
    from ..api.services import LocalServices

CODEX_MODELS: list[HarnessModelInfo] = [
    HarnessModelInfo(
        id="gpt-5.3-codex",
        label="GPT-5.3 Codex",
        description="Default coding model for multi-file edits and tests.",
        options={"effort": ["low", "medium", "high", "xhigh"]},
    ),
    HarnessModelInfo(
        id="gpt-5.2-codex",
        label="GPT-5.2 Codex",
        description="Fast coding model for focused edits.",
        options={"effort": ["low", "medium", "high"]},
    ),
    HarnessModelInfo(
        id="o3",
        label="o3",
        description="Deep reasoning model for tricky bugs and architecture.",
        options={"effort": ["medium", "high"]},
    ),
]


def _find_codex_binary() -> str | None:
    override = os.environ.get("KADAN_CODEX_BIN", "").strip()
    if override and Path(override).is_file():
        return override
    return shutil.which("codex")


def _local_codex_auth_path() -> Path | None:
    codex_home = os.environ.get("CODEX_HOME", "").strip()
    candidates = [
        Path(codex_home) / "auth.json" if codex_home else None,
        Path.home() / ".codex" / "auth.json",
    ]
    for path in candidates:
        if path is not None and path.is_file():
            return path
    return None


def _approval_policy(permission_mode: str) -> str:
    if permission_mode == "full_auto":
        return "never"
    if permission_mode == "auto_edit":
        return "on-failure"
    return "untrusted"


class CodexHarnessSession(HarnessSession):
    """Live `codex app-server` coding session bound to a workspace directory."""

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
        self._home: Path | None = None
        self._rpc: JsonRpcStdioClient | None = None
        self._thread_id: str | None = config.harness_session_id
        self._active_turn_id: str | None = None
        self._pending_server_requests: dict[str, tuple[int | str, str]] = {}

    @property
    def session_id(self) -> str | None:
        return self._thread_id

    async def _ensure_started(self) -> JsonRpcStdioClient:
        if self._rpc is not None:
            return self._rpc
        env = dict(os.environ)
        env.update(self.config.env_overrides)
        if self._credential:
            self._home = Path(tempfile.mkdtemp(prefix="kadan-codex-work-"))
            (self._home / "auth.json").write_text(self._credential)
            env["CODEX_HOME"] = str(self._home)
            env.pop("OPENAI_API_KEY", None)

        cmd = [self.binary, "app-server", "--listen", "stdio://"]
        if self._process_factory is not None:
            transport = await self._process_factory(cmd, self.config.working_dir, env)
        else:
            transport = await StdioProcess.spawn(
                cmd,
                cwd=self.config.working_dir,
                env=env,
                not_found_message="Codex is not installed on this server.",
            )
        rpc = JsonRpcStdioClient(transport, include_jsonrpc=False)
        rpc.start()
        self._rpc = rpc

        await rpc.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "kadan",
                    "title": "Kadan Work",
                    "version": "1.0",
                },
                "capabilities": {"experimentalApi": True},
            },
        )
        await rpc.notify("initialized", {})

        thread_params: dict[str, Any] = {
            "cwd": self.config.working_dir,
            "approvalPolicy": _approval_policy(self.config.permission_mode),
        }
        if self.config.model:
            thread_params["model"] = self.config.model
        effort = self.config.model_options.get("effort")
        if isinstance(effort, str) and effort:
            thread_params["effort"] = effort

        if self._thread_id:
            try:
                resumed = await rpc.request(
                    "thread/resume",
                    {"threadId": self._thread_id, **thread_params},
                )
                self._thread_id = (
                    (resumed.get("thread") or {}).get("id")
                    if isinstance(resumed, dict)
                    else self._thread_id
                ) or self._thread_id
            except HarnessError:
                started = await rpc.request("thread/start", thread_params)
                self._thread_id = (started.get("thread") or {}).get("id")
        else:
            started = await rpc.request("thread/start", thread_params)
            self._thread_id = (started.get("thread") or {}).get("id")

        if not self._thread_id:
            raise HarnessError("Codex did not return a thread identifier.")
        return rpc

    async def start_turn(self, prompt: str) -> AsyncIterator[NormalizedEvent]:
        rpc = await self._ensure_started()
        assert self._thread_id is not None
        yield NormalizedEvent(kind="session_started", session_id=self._thread_id)

        started = await rpc.request(
            "turn/start",
            {
                "threadId": self._thread_id,
                "input": [{"type": "text", "text": prompt}],
            },
        )
        turn_id = (
            (started.get("turn") or {}).get("id") if isinstance(started, dict) else None
        )
        self._active_turn_id = turn_id

        final_text = ""
        while True:
            frame = await rpc.next_message()
            if frame is None:
                detail = await rpc.transport.error_summary()
                yield NormalizedEvent(
                    kind="turn_completed",
                    terminal_status="failed",
                    error=detail,
                )
                return

            method = frame.get("method")
            params = (
                frame.get("params") if isinstance(frame.get("params"), dict) else {}
            )

            # Server-initiated JSON-RPC requests (approvals / user input questions).
            if "id" in frame and isinstance(method, str):
                server_req_id = frame["id"]
                req_key = f"codex-{server_req_id}"
                self._pending_server_requests[req_key] = (server_req_id, method)
                if "requestUserInput" in method or "request_user_input" in method:
                    questions = params.get("questions") or []
                    first_q = (
                        questions[0]
                        if isinstance(questions, list) and questions
                        else {}
                    )
                    prompt_text = (
                        first_q.get("question")
                        or params.get("prompt")
                        or params.get("message")
                        or "Codex has a question before continuing."
                    )
                    options = (
                        first_q.get("options") if isinstance(first_q, dict) else []
                    )
                    yield NormalizedEvent(
                        kind="pending_question",
                        request_id=req_key,
                        title=str(first_q.get("header") or "Question from Codex"),
                        text=str(prompt_text),
                        options=options if isinstance(options, list) else [],
                        raw=params,
                    )
                else:
                    tool_name = str(
                        params.get("tool")
                        or params.get("command")
                        or method.rsplit("/", 1)[-1]
                    )
                    reason = str(
                        params.get("reason")
                        or params.get("message")
                        or params.get("command")
                        or f"Allow Codex to perform {tool_name}?"
                    )
                    yield NormalizedEvent(
                        kind="pending_approval",
                        request_id=req_key,
                        title="Approval requested",
                        text=reason,
                        tool_name=tool_name,
                        arguments=params,
                        options=[
                            {"label": "Approve", "value": "accept"},
                            {
                                "label": "Approve for session",
                                "value": "acceptForSession",
                            },
                            {"label": "Decline", "value": "decline"},
                        ],
                        raw=params,
                    )
                continue

            if not isinstance(method, str):
                continue

            # Filter to our thread when threadId is present.
            if params.get("threadId") and params.get("threadId") != self._thread_id:
                continue

            if method in {
                "item/agentMessage/delta",
                "codex/event/agent_message_delta",
            }:
                delta = str(params.get("delta") or params.get("text") or "")
                if delta:
                    final_text += delta
                    yield NormalizedEvent(kind="text_delta", text=delta, raw=params)
            elif method in {
                "item/reasoning/summaryTextDelta",
                "item/reasoning/textDelta",
                "codex/event/reasoning_delta",
            }:
                delta = str(params.get("delta") or params.get("text") or "")
                if delta:
                    yield NormalizedEvent(
                        kind="reasoning_delta", text=delta, raw=params
                    )
            elif method == "item/started":
                item = (
                    params.get("item") if isinstance(params.get("item"), dict) else {}
                )
                item_type = str(item.get("type") or "")
                if item_type in {"commandExecution", "mcpToolCall", "dynamicToolCall"}:
                    yield NormalizedEvent(
                        kind="tool_call",
                        call_id=str(item.get("id") or ""),
                        tool_name=str(
                            item.get("tool") or item.get("command") or item_type
                        ),
                        arguments={
                            k: v
                            for k, v in item.items()
                            if k in {"command", "arguments", "cwd"}
                        },
                        raw=item,
                    )
            elif method == "item/completed":
                item = (
                    params.get("item") if isinstance(params.get("item"), dict) else {}
                )
                item_type = str(item.get("type") or "")
                if item_type == "agentMessage":
                    text = str(item.get("text") or "")
                    if text:
                        final_text = text
                        yield NormalizedEvent(
                            kind="assistant_message", text=text, raw=item
                        )
                elif item_type == "fileChange":
                    changes = item.get("changes") or []
                    if isinstance(changes, list) and changes:
                        for change in changes:
                            if isinstance(change, dict):
                                yield NormalizedEvent(
                                    kind="file_change",
                                    file_path=str(change.get("path") or ""),
                                    change_type=str(change.get("kind") or "modified"),
                                    text=str(change.get("diff") or "")[:4000],
                                    raw=change,
                                )
                    else:
                        yield NormalizedEvent(
                            kind="file_change",
                            file_path=str(item.get("path") or ""),
                            change_type=str(item.get("kind") or "modified"),
                            raw=item,
                        )
                elif item_type in {
                    "commandExecution",
                    "mcpToolCall",
                    "dynamicToolCall",
                }:
                    output = (
                        item.get("aggregatedOutput")
                        or item.get("output")
                        or item.get("result")
                        or ""
                    )
                    if not isinstance(output, str):
                        output = json.dumps(output, ensure_ascii=False)
                    exit_code = item.get("exitCode")
                    is_err = bool(item.get("isError")) or (
                        isinstance(exit_code, int) and exit_code != 0
                    )
                    yield NormalizedEvent(
                        kind="tool_result",
                        call_id=str(item.get("id") or ""),
                        tool_name=str(
                            item.get("tool") or item.get("command") or item_type
                        ),
                        text=output[:8000],
                        is_error=is_err,
                        raw=item,
                    )
            elif method == "error" and not params.get("willRetry"):
                err_obj = params.get("error")
                msg = (
                    err_obj.get("message")
                    if isinstance(err_obj, dict) and err_obj.get("message")
                    else str(err_obj or "Codex reported an error.")
                )
                yield NormalizedEvent(kind="error", text=str(msg), error=str(msg))
            elif method == "turn/completed":
                turn = (
                    params.get("turn") if isinstance(params.get("turn"), dict) else {}
                )
                if turn_id and turn.get("id") and turn.get("id") != turn_id:
                    continue
                self._active_turn_id = None
                raw_status = str(turn.get("status") or "completed").lower()
                if raw_status == "interrupted":
                    terminal = "interrupted"
                elif raw_status in {"cancelled", "canceled"}:
                    terminal = "cancelled"
                elif raw_status != "completed":
                    terminal = "failed"
                else:
                    terminal = "completed"
                err_msg = None
                if isinstance(turn.get("error"), dict):
                    err_msg = str(turn["error"].get("message") or raw_status)
                elif terminal == "failed":
                    err_msg = raw_status
                yield NormalizedEvent(
                    kind="turn_completed",
                    text=final_text,
                    terminal_status=terminal,
                    error=err_msg,
                    raw=turn,
                )
                return

    async def steer(self, prompt: str) -> bool:
        if self._rpc is None or not self._thread_id or not self._active_turn_id:
            return False
        try:
            await self._rpc.request(
                "turn/steer",
                {
                    "threadId": self._thread_id,
                    "expectedTurnId": self._active_turn_id,
                    "input": [{"type": "text", "text": prompt}],
                },
                timeout=10.0,
            )
            return True
        except HarnessError:
            return False

    async def interrupt(self) -> None:
        if self._rpc is None or not self._thread_id:
            return
        with suppress(HarnessError, Exception):
            await self._rpc.request(
                "turn/interrupt",
                {
                    "threadId": self._thread_id,
                    **(
                        {"turnId": self._active_turn_id} if self._active_turn_id else {}
                    ),
                },
                timeout=5.0,
            )

    async def respond_pending(self, request_id: str, response: dict[str, Any]) -> None:
        if self._rpc is None:
            raise HarnessError("Codex session is not running.")
        entry = self._pending_server_requests.pop(request_id, None)
        if entry is None:
            raise HarnessError("That pending Codex request is no longer active.")
        server_req_id, method = entry
        if "requestUserInput" in method or "request_user_input" in method:
            answers = response.get("answers")
            if answers is None:
                text = str(response.get("answer") or response.get("text") or "")
                answers = {"answer": {"answers": [text]}} if text else {}
            await self._rpc.respond(server_req_id, result={"answers": answers})
        else:
            decision = str(response.get("decision") or "decline").lower()
            mapped = {
                "approve": "accept",
                "accept": "accept",
                "allow": "accept",
                "acceptforsession": "acceptForSession",
                "approve_session": "acceptForSession",
                "deny": "decline",
                "decline": "decline",
            }.get(decision, "decline")
            await self._rpc.respond(server_req_id, result={"decision": mapped})

    async def close(self) -> None:
        if self._rpc is not None:
            await self._rpc.close()
            self._rpc = None
        if self._home is not None:
            shutil.rmtree(self._home, ignore_errors=True)
            self._home = None


class CodexHarnessAdapter(HarnessAdapter):
    """Bundled ChatGPT Codex adapter using `codex app-server`."""

    harness_id = "codex"
    display_name = "Codex"
    description = "OpenAI's coding agent using your ChatGPT Codex sign-in."

    def __init__(self, *, process_factory: ProcessFactory | None = None):
        self._process_factory = process_factory

    async def _stored_credential(
        self, services: LocalServices
    ) -> tuple[str | None, str | None]:
        try:
            cfg = await services.json_request(
                "Tsuchi", "GET", "/api/work/harnesses/codex"
            )
        except ServiceError:
            cfg = None
        if isinstance(cfg, dict) and cfg.get("credential"):
            return str(cfg["credential"]), cfg.get("account_label") or "ChatGPT Codex"
        local = _local_codex_auth_path()
        if local is not None:
            with suppress(OSError):
                text = local.read_text().strip()
                if text:
                    return text, "Local Codex sign-in"
        return None, None

    async def _hikari_codex_connected(self, services: LocalServices) -> bool:
        try:
            subs = await services.json_request("Hikari", "GET", "/v1/subscriptions")
        except ServiceError:
            return False
        if isinstance(subs, dict):
            for entry in subs.get("subscriptions") or []:
                if (
                    isinstance(entry, dict)
                    and entry.get("provider") == "codex"
                    and entry.get("connected")
                ):
                    return True
        return False

    async def check_availability(self, services: LocalServices) -> HarnessAvailability:
        binary = _find_codex_binary()
        in_flight = cli_logins.status("codex")
        cred, label = await self._stored_credential(services)
        if not binary:
            auth = HarnessAuthStatus(
                status="unavailable",
                login_method="device_code",
                message="Codex is not installed on this server yet.",
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
                login_method="device_code",
                account_label=label,
                message=f"Connected ({label}).",
            )
        else:
            hikari_ok = await self._hikari_codex_connected(services)
            auth = HarnessAuthStatus(
                status="sign_in_required",
                login_method="device_code",
                message=(
                    "Sign in with ChatGPT Codex to run coding threads."
                    if not hikari_ok
                    else "Sign in once here to authorize Codex for coding workspaces."
                ),
            )
        return HarnessAvailability(
            harness_id=self.harness_id,
            display_name=self.display_name,
            description=self.description,
            installed=binary is not None,
            binary_path=binary,
            auth=auth,
            models=CODEX_MODELS,
            default_model=CODEX_MODELS[0].id,
            supports_steer=True,
            supports_resume=True,
            supports_approvals=True,
            supports_questions=True,
        )

    async def start_sign_in(
        self, services: LocalServices, payload: dict[str, Any] | None = None
    ) -> HarnessAuthStatus:
        binary = _find_codex_binary()
        if not binary:
            raise HarnessError("Codex is not installed on this server.")
        return await cli_logins.start("codex", binary, services)

    async def sign_in_status(self, services: LocalServices) -> HarnessAuthStatus:
        in_flight = cli_logins.status("codex")
        if in_flight is not None:
            return in_flight
        return (await self.check_availability(services)).auth

    async def sign_out(self, services: LocalServices) -> HarnessAuthStatus:
        await cli_logins.cancel("codex")
        with suppress(ServiceError):
            await services.json_request("Tsuchi", "DELETE", "/api/work/harnesses/codex")
        return HarnessAuthStatus(
            status="sign_in_required",
            login_method="device_code",
            message="Signed out of Codex.",
        )

    async def open_session(
        self, config: SessionConfig, services: LocalServices
    ) -> HarnessSession:
        binary = _find_codex_binary() or "codex"
        if self._process_factory is None and not _find_codex_binary():
            raise HarnessError("Codex is not installed on this server.")
        cred, _ = await self._stored_credential(services)
        return CodexHarnessSession(
            config,
            binary=binary,
            credential=cred,
            process_factory=self._process_factory,
        )
