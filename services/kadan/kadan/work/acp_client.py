"""Shared Agent Client Protocol (ACP) stdio JSON-RPC client with v2 -> v1 fallback.

Powers both Google Antigravity (`antigravity-acp`) and xAI Grok (`grok` ACP mode)
via a single clean wire implementation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
from typing import Any

from .contract import (
    HarnessError,
    HarnessModelInfo,
    HarnessSession,
    NormalizedEvent,
    SessionConfig,
)
from .stdio_jsonrpc import JsonRpcStdioClient, ProcessFactory, StdioProcess

logger = logging.getLogger(__name__)

ACP_V2_VERSION = 2
ACP_V1_VERSION = 1

_EFFORT_KEY_RE = re.compile(r"effort|reasoning|thought", re.IGNORECASE)


def _flatten_acp_config_values(raw_options: Any) -> list[dict[str, Any]]:
    """Flatten ACP v2 `configOptions[].options`, which may be flat or grouped."""
    if not isinstance(raw_options, list):
        return []
    flat: list[dict[str, Any]] = []
    for item in raw_options:
        if not isinstance(item, dict):
            continue
        nested = item.get("options")
        if isinstance(nested, list):
            for sub in nested:
                if isinstance(sub, dict):
                    flat.append(sub)
        else:
            flat.append(item)
    return flat


def _extract_model_efforts(entry: dict[str, Any]) -> list[str]:
    for key in ("supportedEffortLevels", "effortLevels", "reasoningEfforts"):
        raw = entry.get(key)
        if isinstance(raw, list):
            levels: list[str] = []
            for item in raw:
                val = (
                    item.get("value") or item.get("reasoningEffort") or item.get("id")
                    if isinstance(item, dict)
                    else item
                )
                if isinstance(val, str) and val.strip() and val.strip() not in levels:
                    levels.append(val.strip())
            if levels:
                return levels
    return []


def parse_acp_models(
    init_res: dict[str, Any] | None,
    session_res: dict[str, Any] | None,
) -> tuple[list[HarnessModelInfo], str | None]:
    """Extract selectable models and reasoning levels from ACP `initialize`/`session/new` responses."""
    session = session_res if isinstance(session_res, dict) else {}
    init = init_res if isinstance(init_res, dict) else {}

    config_options = (
        session.get("configOptions")
        if isinstance(session.get("configOptions"), list)
        else []
    )

    shared_efforts: list[str] = []
    for opt in config_options:
        if not isinstance(opt, dict):
            continue
        cat = str(opt.get("category") or "")
        opt_id = str(opt.get("id") or "")
        if _EFFORT_KEY_RE.search(cat) or _EFFORT_KEY_RE.search(opt_id):
            for choice in _flatten_acp_config_values(opt.get("options")):
                val = str(choice.get("value") or choice.get("id") or "").strip()
                if val and val not in shared_efforts:
                    shared_efforts.append(val)
            if shared_efforts:
                break

    models: list[HarnessModelInfo] = []
    seen: set[str] = set()
    default_model: str | None = None

    def _add_model(
        model_id: str,
        label: str,
        description: str,
        efforts: list[str],
    ) -> None:
        mid = model_id.strip()
        if not mid or mid in seen:
            return
        seen.add(mid)
        opts = {"effort": efforts} if efforts else {}
        models.append(
            HarnessModelInfo(
                id=mid,
                label=label.strip() or mid,
                description=description.strip(),
                options=opts,
            )
        )

    # 1. Standard ACP `models: { currentModelId, availableModels: [...] }`.
    models_block = session.get("models")
    if isinstance(models_block, dict):
        cur = models_block.get("currentModelId") or models_block.get("current_model_id")
        if isinstance(cur, str) and cur.strip():
            default_model = cur.strip()
        avail = (
            models_block.get("availableModels")
            or models_block.get("available_models")
            or models_block.get("models")
        )
        if isinstance(avail, list):
            for entry in avail:
                if not isinstance(entry, dict):
                    continue
                mid = str(
                    entry.get("modelId")
                    or entry.get("model_id")
                    or entry.get("id")
                    or ""
                )
                label = str(
                    entry.get("name")
                    or entry.get("displayName")
                    or entry.get("label")
                    or mid
                )
                desc = str(entry.get("description") or "")
                efforts = _extract_model_efforts(entry) or list(shared_efforts)
                _add_model(mid, label, desc, efforts)

    # 2. ACP v2 `configOptions` with `category == "model"` or `id == "model"`.
    if not models:
        for opt in config_options:
            if not isinstance(opt, dict):
                continue
            cat = str(opt.get("category") or "").lower()
            opt_id = str(opt.get("id") or "").lower()
            if cat != "model" and opt_id != "model":
                continue
            cur = opt.get("currentValue") or opt.get("current_value")
            if isinstance(cur, str) and cur.strip():
                default_model = cur.strip()
            for choice in _flatten_acp_config_values(opt.get("options")):
                mid = str(choice.get("value") or choice.get("id") or "")
                label = str(choice.get("name") or choice.get("label") or mid)
                desc = str(choice.get("description") or "")
                efforts = _extract_model_efforts(choice) or list(shared_efforts)
                _add_model(mid, label, desc, efforts)
            if models:
                break

    # 3. Fallback: `initialize` result `models`.
    if not models:
        init_models = init.get("models")
        raw_list: list[Any] = []
        if isinstance(init_models, dict):
            cur = init_models.get("currentModelId") or init_models.get("defaultModel")
            if isinstance(cur, str) and cur.strip():
                default_model = cur.strip()
            cand = (
                init_models.get("availableModels")
                or init_models.get("models")
                or init_models.get("data")
            )
            if isinstance(cand, list):
                raw_list = cand
        elif isinstance(init_models, list):
            raw_list = init_models
        for entry in raw_list:
            if isinstance(entry, str):
                _add_model(entry, entry, "", list(shared_efforts))
            elif isinstance(entry, dict):
                mid = str(
                    entry.get("modelId") or entry.get("id") or entry.get("model") or ""
                )
                label = str(
                    entry.get("name")
                    or entry.get("displayName")
                    or entry.get("label")
                    or mid
                )
                desc = str(entry.get("description") or "")
                efforts = _extract_model_efforts(entry) or list(shared_efforts)
                _add_model(mid, label, desc, efforts)

    if default_model and default_model not in seen:
        default_model = models[0].id if models else None
    elif not default_model and models:
        default_model = models[0].id

    return models, default_model


async def discover_acp_models(
    command: list[str],
    *,
    display_name: str,
    env_extra: dict[str, str] | None = None,
    cwd: str | None = None,
    process_factory: ProcessFactory | None = None,
    timeout: float = 15.0,
) -> tuple[list[HarnessModelInfo], str | None]:
    """Probe an ACP server via `initialize` + `session/new` to discover live models."""
    work_dir = cwd or os.getcwd()
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    rpc: JsonRpcStdioClient | None = None
    session_id: str | None = None
    try:
        if process_factory is not None:
            transport = await process_factory(command, work_dir, env)
        else:
            transport = await StdioProcess.spawn(
                command,
                cwd=work_dir,
                env=env,
                not_found_message=f"{display_name} is not installed on this computer.",
            )
        rpc = JsonRpcStdioClient(transport, include_jsonrpc=True)
        rpc.start()

        client_info = {"name": "kadan", "title": "Kadan Work", "version": "1.0"}
        try:
            init_res = await rpc.request(
                "initialize",
                {
                    "protocolVersion": ACP_V2_VERSION,
                    "clientInfo": client_info,
                    "clientCapabilities": {},
                },
                timeout=timeout,
            )
        except HarnessError:
            init_res = await rpc.request(
                "initialize",
                {
                    "protocolVersion": ACP_V1_VERSION,
                    "clientInfo": client_info,
                    "clientCapabilities": {},
                },
                timeout=timeout,
            )

        session_res: dict[str, Any] = {}
        try:
            created = await rpc.request(
                "session/new",
                {"cwd": work_dir, "mcpServers": []},
                timeout=timeout,
            )
            if isinstance(created, dict):
                session_res = created
                sid = created.get("sessionId")
                if isinstance(sid, str) and sid:
                    session_id = sid
        except HarnessError as exc:
            logger.debug("%s ACP session/new model probe failed: %s", display_name, exc)

        return parse_acp_models(
            init_res if isinstance(init_res, dict) else {},
            session_res,
        )
    except (HarnessError, TimeoutError, OSError) as exc:
        logger.warning("%s ACP model discovery failed: %s", display_name, exc)
        return [], None
    finally:
        if rpc is not None:
            if session_id:
                with suppress(Exception):
                    await rpc.request(
                        "session/close", {"sessionId": session_id}, timeout=2.0
                    )
            with suppress(Exception):
                await rpc.close()


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        inner = content.get("content")
        if inner is not None:
            return _extract_text_content(inner)
    if isinstance(content, list):
        parts = [_extract_text_content(item) for item in content]
        return "".join(part for part in parts if part)
    return ""


class AcpHarnessSession(HarnessSession):
    """Live ACP coding session negotiating ACP v2 with automatic v1 fallback."""

    def __init__(
        self,
        config: SessionConfig,
        *,
        command: list[str],
        display_name: str,
        env_extra: dict[str, str] | None = None,
        process_factory: ProcessFactory | None = None,
    ):
        self.config = config
        self.command = command
        self.display_name = display_name
        self._env_extra = env_extra or {}
        self._process_factory = process_factory
        self._rpc: JsonRpcStdioClient | None = None
        self._session_id: str | None = config.harness_session_id
        self.protocol_version: int = ACP_V2_VERSION
        self._pending_requests: dict[
            str, tuple[int | str, str, list[dict[str, Any]]]
        ] = {}
        self._cancelled = False

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def _initialize_protocol(self, rpc: JsonRpcStdioClient) -> dict[str, Any]:
        """Negotiate ACP v2 first; automatically fall back to ACP v1 if rejected."""
        client_info = {"name": "kadan", "title": "Kadan Work", "version": "1.0"}
        v2_params = {
            "protocolVersion": ACP_V2_VERSION,
            "clientInfo": client_info,
            "clientCapabilities": {
                "fs": {"readTextFile": True, "writeTextFile": True},
                "terminal": False,
                "userInput": True,
            },
        }
        try:
            init_res = await rpc.request("initialize", v2_params, timeout=30.0)
            negotiated = (
                init_res.get("protocolVersion") if isinstance(init_res, dict) else None
            )
            if negotiated in {1, "1", "v1"}:
                self.protocol_version = ACP_V1_VERSION
            else:
                self.protocol_version = ACP_V2_VERSION
            return init_res if isinstance(init_res, dict) else {}
        except HarnessError:
            # Server rejected v2 handshake: retry using ACP v1 payload.
            v1_params = {
                "protocolVersion": ACP_V1_VERSION,
                "clientInfo": client_info,
                "clientCapabilities": {
                    "fs": {"readTextFile": True, "writeTextFile": True},
                },
            }
            init_res = await rpc.request("initialize", v1_params, timeout=30.0)
            self.protocol_version = ACP_V1_VERSION
            return init_res if isinstance(init_res, dict) else {}

    async def _ensure_started(self) -> JsonRpcStdioClient:
        if self._rpc is not None:
            return self._rpc
        env = dict(os.environ)
        env.update(self._env_extra)
        env.update(self.config.env_overrides)
        if self._process_factory is not None:
            transport = await self._process_factory(
                self.command, self.config.working_dir, env
            )
        else:
            transport = await StdioProcess.spawn(
                self.command,
                cwd=self.config.working_dir,
                env=env,
                not_found_message=f"{self.display_name} is not installed on this server.",
            )
        rpc = JsonRpcStdioClient(transport, include_jsonrpc=True)
        rpc.start()
        self._rpc = rpc

        await self._initialize_protocol(rpc)

        base_session_params: dict[str, Any] = {
            "cwd": self.config.working_dir,
            "mcpServers": [],
        }
        if self.config.model:
            base_session_params["model"] = self.config.model

        if self._session_id:
            load_method = (
                "session/resume"
                if self.protocol_version >= ACP_V2_VERSION
                else "session/load"
            )
            try:
                loaded = await rpc.request(
                    load_method,
                    {"sessionId": self._session_id, **base_session_params},
                    timeout=30.0,
                )
                if isinstance(loaded, dict) and loaded.get("sessionId"):
                    self._session_id = str(loaded["sessionId"])
            except HarnessError:
                created = await rpc.request(
                    "session/new", base_session_params, timeout=30.0
                )
                self._session_id = (
                    str(created.get("sessionId") or "")
                    if isinstance(created, dict)
                    else ""
                )
        else:
            created = await rpc.request(
                "session/new", base_session_params, timeout=30.0
            )
            self._session_id = (
                str(created.get("sessionId") or "") if isinstance(created, dict) else ""
            )

        if not self._session_id:
            raise HarnessError(
                f"{self.display_name} did not return a session identifier."
            )

        # Apply selected model via session/set_model when supported (best effort).
        if self.config.model:
            with suppress(HarnessError):
                await rpc.request(
                    "session/set_model",
                    {"sessionId": self._session_id, "modelId": self.config.model},
                    timeout=10.0,
                )
        return rpc

    async def _handle_fs_request(
        self,
        rpc: JsonRpcStdioClient,
        req_id: int | str,
        method: str,
        params: dict[str, Any],
    ) -> None:
        raw_path = str(params.get("path") or "")
        if not raw_path:
            await rpc.respond(
                req_id, error={"code": -32602, "message": "Missing file path."}
            )
            return
        target = Path(raw_path)
        if not target.is_absolute():
            target = Path(self.config.working_dir) / target
        try:
            if method == "fs/read_text_file":
                content = await asyncio.to_thread(target.read_text, encoding="utf-8")
                await rpc.respond(req_id, result={"content": content})
            elif method == "fs/write_text_file":
                target.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(
                    target.write_text,
                    str(params.get("content") or ""),
                    encoding="utf-8",
                )
                await rpc.respond(req_id, result={})
        except OSError as exc:
            await rpc.respond(req_id, error={"code": -32000, "message": str(exc)})

    def _translate_update(self, update: dict[str, Any]) -> list[NormalizedEvent]:
        kind = str(
            update.get("sessionUpdate")
            or update.get("type")
            or update.get("kind")
            or ""
        )
        events: list[NormalizedEvent] = []
        if kind in {"agent_message_chunk", "message_delta", "text_delta"}:
            chunk = _extract_text_content(
                update.get("content") or update.get("delta") or update.get("text")
            )
            if chunk:
                events.append(
                    NormalizedEvent(kind="text_delta", text=chunk, raw=update)
                )
        elif kind in {"agent_thought_chunk", "thought_delta", "reasoning_delta"}:
            chunk = _extract_text_content(
                update.get("content") or update.get("delta") or update.get("text")
            )
            if chunk:
                events.append(
                    NormalizedEvent(kind="reasoning_delta", text=chunk, raw=update)
                )
        elif kind == "tool_call":
            events.append(
                NormalizedEvent(
                    kind="tool_call",
                    call_id=str(update.get("toolCallId") or update.get("id") or ""),
                    tool_name=str(
                        update.get("title")
                        or update.get("name")
                        or update.get("kind")
                        or "tool"
                    ),
                    arguments=update.get("rawInput")
                    if isinstance(update.get("rawInput"), dict)
                    else {},
                    raw=update,
                )
            )
        elif kind == "tool_call_update":
            status = str(update.get("status") or "").lower()
            raw_out = update.get("rawOutput") or update.get("content") or ""
            text = _extract_text_content(raw_out)
            if not text and raw_out:
                text = json.dumps(raw_out, ensure_ascii=False)
            if status in {"completed", "failed", "error"} or text:
                events.append(
                    NormalizedEvent(
                        kind="tool_result",
                        call_id=str(update.get("toolCallId") or update.get("id") or ""),
                        tool_name=str(
                            update.get("title") or update.get("name") or "tool"
                        ),
                        text=text[:8000],
                        is_error=status in {"failed", "error"},
                        raw=update,
                    )
                )
            for loc in update.get("locations") or []:
                if isinstance(loc, dict) and isinstance(loc.get("path"), str):
                    events.append(
                        NormalizedEvent(
                            kind="file_change",
                            file_path=loc["path"],
                            change_type="modified",
                            raw=loc,
                        )
                    )
        elif kind in {"file_change", "diff"}:
            events.append(
                NormalizedEvent(
                    kind="file_change",
                    file_path=str(update.get("path") or ""),
                    change_type=str(update.get("changeType") or "modified"),
                    text=str(update.get("diff") or "")[:4000],
                    raw=update,
                )
            )
        elif kind in {"plan", "status", "session_info_update"}:
            summary = str(update.get("title") or update.get("message") or "")
            if summary:
                events.append(NormalizedEvent(kind="status", text=summary, raw=update))
        return events

    async def start_turn(self, prompt: str) -> AsyncIterator[NormalizedEvent]:
        self._cancelled = False
        rpc = await self._ensure_started()
        assert self._session_id is not None
        yield NormalizedEvent(kind="session_started", session_id=self._session_id)

        prompt_task = asyncio.create_task(
            rpc.request(
                "session/prompt",
                {
                    "sessionId": self._session_id,
                    "prompt": [{"type": "text", "text": prompt}],
                },
                timeout=3600.0,
            )
        )

        final_parts: list[str] = []
        try:
            while True:
                msg_task = asyncio.create_task(rpc.next_message())
                done, _ = await asyncio.wait(
                    {prompt_task, msg_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if msg_task in done:
                    frame = msg_task.result()
                    if frame is None:
                        if prompt_task.done():
                            break
                        detail = await rpc.transport.error_summary()
                        prompt_task.cancel()
                        yield NormalizedEvent(
                            kind="turn_completed",
                            text="".join(final_parts),
                            terminal_status="failed",
                            error=detail,
                        )
                        return
                    method = frame.get("method")
                    params = (
                        frame.get("params")
                        if isinstance(frame.get("params"), dict)
                        else {}
                    )
                    if "id" in frame and isinstance(method, str):
                        req_id = frame["id"]
                        if method in {"fs/read_text_file", "fs/write_text_file"}:
                            await self._handle_fs_request(rpc, req_id, method, params)
                            continue
                        req_key = f"acp-{req_id}"
                        options = (
                            params.get("options")
                            if isinstance(params.get("options"), list)
                            else []
                        )
                        self._pending_requests[req_key] = (req_id, method, options)
                        if method in {
                            "session/request_user_input",
                            "session/ask_user",
                            "user/ask",
                        }:
                            yield NormalizedEvent(
                                kind="pending_question",
                                request_id=req_key,
                                title=str(
                                    params.get("title")
                                    or f"Question from {self.display_name}"
                                ),
                                text=str(
                                    params.get("question")
                                    or params.get("prompt")
                                    or f"{self.display_name} has a question."
                                ),
                                options=options,
                                raw=params,
                            )
                        else:
                            tool_call = (
                                params.get("toolCall")
                                if isinstance(params.get("toolCall"), dict)
                                else {}
                            )
                            tool_name = str(
                                tool_call.get("title")
                                or tool_call.get("name")
                                or params.get("toolName")
                                or "action"
                            )
                            formatted_options = [
                                {
                                    "label": str(
                                        opt.get("name")
                                        or opt.get("label")
                                        or opt.get("optionId")
                                        or "Option"
                                    ),
                                    "value": str(
                                        opt.get("optionId")
                                        or opt.get("id")
                                        or opt.get("kind")
                                        or ""
                                    ),
                                    "kind": opt.get("kind"),
                                }
                                for opt in options
                                if isinstance(opt, dict)
                            ] or [
                                {"label": "Approve", "value": "allow_once"},
                                {"label": "Decline", "value": "reject_once"},
                            ]
                            yield NormalizedEvent(
                                kind="pending_approval",
                                request_id=req_key,
                                title=f"Allow {tool_name}?",
                                text=str(
                                    params.get("message")
                                    or tool_call.get("title")
                                    or f"Allow {self.display_name} to run {tool_name}?"
                                ),
                                tool_name=tool_name,
                                arguments=tool_call.get("rawInput")
                                if isinstance(tool_call.get("rawInput"), dict)
                                else params,
                                options=formatted_options,
                                raw=params,
                            )
                    elif method == "session/update":
                        update = (
                            params.get("update")
                            if isinstance(params.get("update"), dict)
                            else params
                        )
                        for ev in self._translate_update(update):
                            if ev.kind == "text_delta":
                                final_parts.append(ev.text)
                            yield ev
                else:
                    msg_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await msg_task
                    break

            # Drain any remaining queued session/update frames before finishing.
            while not rpc.incoming.empty():
                trailing = rpc.incoming.get_nowait()
                if trailing and trailing.get("method") == "session/update":
                    tparams = trailing.get("params") or {}
                    update = (
                        tparams.get("update")
                        if isinstance(tparams.get("update"), dict)
                        else tparams
                    )
                    for ev in self._translate_update(update):
                        if ev.kind == "text_delta":
                            final_parts.append(ev.text)
                        yield ev

            result = await prompt_task
            stop_reason = (
                str(result.get("stopReason") or "end_turn").lower()
                if isinstance(result, dict)
                else "end_turn"
            )
            if self._cancelled or stop_reason in {"cancelled", "canceled"}:
                terminal = "interrupted"
            elif stop_reason in {"error", "failed"}:
                terminal = "failed"
            else:
                terminal = "completed"
            yield NormalizedEvent(
                kind="turn_completed",
                text="".join(final_parts),
                terminal_status=terminal,
                raw=result if isinstance(result, dict) else {},
            )
        except HarnessError as exc:
            yield NormalizedEvent(
                kind="turn_completed",
                text="".join(final_parts),
                terminal_status="interrupted" if self._cancelled else "failed",
                error=None if self._cancelled else str(exc),
            )

    async def interrupt(self) -> None:
        self._cancelled = True
        if self._rpc is None or not self._session_id:
            return
        with suppress(Exception):
            await self._rpc.notify("session/cancel", {"sessionId": self._session_id})

    async def respond_pending(self, request_id: str, response: dict[str, Any]) -> None:
        if self._rpc is None:
            raise HarnessError(f"{self.display_name} session is not running.")
        entry = self._pending_requests.pop(request_id, None)
        if entry is None:
            raise HarnessError("That pending request is no longer active.")
        rpc_id, method, options = entry
        if method in {"session/request_user_input", "session/ask_user", "user/ask"}:
            await self._rpc.respond(
                rpc_id,
                result={
                    "answer": str(response.get("answer") or response.get("text") or ""),
                    "answers": response.get("answers"),
                },
            )
            return

        decision = str(
            response.get("option_id")
            or response.get("optionId")
            or response.get("decision")
            or "reject_once"
        )
        chosen_id: str | None = None
        if decision.lower() in {"approve", "allow", "accept", "allow_once"}:
            for opt in options:
                if isinstance(opt, dict) and str(opt.get("kind") or "").startswith(
                    "allow"
                ):
                    chosen_id = str(opt.get("optionId") or opt.get("id") or "")
                    break
            chosen_id = chosen_id or "allow_once"
        elif decision.lower() in {"deny", "decline", "reject", "reject_once"}:
            for opt in options:
                if isinstance(opt, dict) and str(opt.get("kind") or "").startswith(
                    "reject"
                ):
                    chosen_id = str(opt.get("optionId") or opt.get("id") or "")
                    break
            chosen_id = chosen_id or "reject_once"
        else:
            chosen_id = decision

        await self._rpc.respond(
            rpc_id,
            result={"outcome": {"outcome": "selected", "optionId": chosen_id}},
        )

    async def close(self) -> None:
        if self._rpc is not None:
            await self._rpc.close()
            self._rpc = None
