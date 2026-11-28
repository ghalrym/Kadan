"""Unit tests for Codex app-server, Claude Code stream-json, and ACP v2/v1 adapters with mocked stdio."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any
from uuid import uuid4

import pytest

from kadan.work.acp_client import (
    ACP_V1_VERSION,
    ACP_V2_VERSION,
    AcpHarnessSession,
    parse_acp_models,
)
from kadan.work.antigravity_adapter import (
    AntigravityHarnessAdapter,
    _parse_agy_models_output,
)
from kadan.work.claude_adapter import (
    ClaudeHarnessAdapter,
    ClaudeHarnessSession,
    _parse_claude_models,
)
from kadan.work.codex_adapter import (
    CodexHarnessAdapter,
    CodexHarnessSession,
    _parse_codex_models,
)
from kadan.work.contract import HarnessModelInfo, ModelCatalogCache, SessionConfig
from kadan.work.grok_adapter import GrokHarnessAdapter, _parse_grok_models_output


class FakeStdioTransport:
    """In-memory bidirectional JSON line transport mimicking `StdioProcess`."""

    def __init__(
        self,
        on_write: Callable[[dict[str, Any], "FakeStdioTransport"], None],
    ):
        self.on_write = on_write
        self.written: list[dict[str, Any]] = []
        self.inbox: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.closed = False

    def push(self, frame: dict[str, Any] | None) -> None:
        self.inbox.put_nowait(frame)

    async def write_json(self, frame: dict[str, Any]) -> None:
        self.written.append(frame)
        self.on_write(frame, self)

    async def read_json(self) -> dict[str, Any] | None:
        return await self.inbox.get()

    async def error_summary(self) -> str:
        return "mock process closed"

    async def close(self) -> None:
        self.closed = True
        self.inbox.put_nowait(None)


@pytest.mark.anyio
async def test_codex_app_server_turn_approval_steer_and_file_change(tmp_path):
    captured_cmd: list[str] = []

    def handle_write(frame: dict[str, Any], transport: FakeStdioTransport) -> None:
        method = frame.get("method")
        req_id = frame.get("id")
        if method == "initialize":
            transport.push({"id": req_id, "result": {"serverInfo": {"name": "codex"}}})
        elif method == "thread/start":
            transport.push({"id": req_id, "result": {"thread": {"id": "thr-codex-1"}}})
        elif method == "turn/start":
            transport.push({"id": req_id, "result": {"turn": {"id": "turn-1"}}})
            transport.push(
                {
                    "method": "item/started",
                    "params": {
                        "threadId": "thr-codex-1",
                        "item": {
                            "id": "cmd-1",
                            "type": "commandExecution",
                            "command": "pytest",
                        },
                    },
                }
            )
            # Server asks for approval.
            transport.push(
                {
                    "id": 900,
                    "method": "item/commandExecution/requestApproval",
                    "params": {"command": "pytest", "reason": "Run unit tests"},
                }
            )
        elif method == "turn/steer":
            transport.push({"id": req_id, "result": {"accepted": True}})
        elif "result" in frame and frame.get("id") == 900:
            # Approval answered: emit tool output, file change, delta, completion.
            assert frame["result"]["decision"] == "accept"
            transport.push(
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": "thr-codex-1",
                        "item": {
                            "id": "cmd-1",
                            "type": "commandExecution",
                            "command": "pytest",
                            "aggregatedOutput": "4 passed",
                            "exitCode": 0,
                        },
                    },
                }
            )
            transport.push(
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": "thr-codex-1",
                        "item": {
                            "id": "fc-1",
                            "type": "fileChange",
                            "changes": [{"path": "app.py", "kind": "modified"}],
                        },
                    },
                }
            )
            transport.push(
                {
                    "method": "item/agentMessage/delta",
                    "params": {"threadId": "thr-codex-1", "delta": "All tests pass."},
                }
            )
            transport.push(
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thr-codex-1",
                        "turn": {"id": "turn-1", "status": "completed"},
                    },
                }
            )

    transport = FakeStdioTransport(handle_write)

    async def factory(cmd, cwd, env):
        captured_cmd.extend(cmd)
        assert cwd == str(tmp_path)
        return transport

    session = CodexHarnessSession(
        SessionConfig(
            thread_id=uuid4(), working_dir=str(tmp_path), model="gpt-5.3-codex"
        ),
        binary="codex",
        process_factory=factory,
    )

    events = []
    steered = False
    async for ev in session.start_turn("Run tests"):
        events.append(ev)
        if ev.kind == "pending_approval":
            steered = await session.steer("Keep output short")
            await session.respond_pending(ev.request_id, {"decision": "approve"})

    await session.close()
    assert captured_cmd[:3] == ["codex", "app-server", "--listen"]
    assert session.session_id == "thr-codex-1"
    assert steered is True
    kinds = [e.kind for e in events]
    assert "session_started" in kinds
    assert "tool_call" in kinds
    assert "pending_approval" in kinds
    assert "tool_result" in kinds
    assert "file_change" in kinds
    assert "text_delta" in kinds
    assert events[-1].kind == "turn_completed"
    assert events[-1].terminal_status == "completed"
    assert events[-1].text == "All tests pass."


@pytest.mark.anyio
async def test_claude_stream_json_question_tool_use_and_resume(tmp_path):
    captured_cmd: list[str] = []

    def handle_write(frame: dict[str, Any], transport: FakeStdioTransport) -> None:
        if frame.get("type") == "user":
            transport.push(
                {"type": "system", "subtype": "init", "session_id": "claude-sess-42"}
            )
            transport.push(
                {
                    "type": "control_request",
                    "request_id": "ctrl-q1",
                    "request": {
                        "subtype": "can_use_tool",
                        "tool_name": "AskUserQuestion",
                        "input": {
                            "questions": [
                                {
                                    "header": "Target branch",
                                    "question": "Which branch should I target?",
                                    "options": [{"label": "main"}, {"label": "dev"}],
                                }
                            ]
                        },
                    },
                }
            )
        elif frame.get("type") == "control_response":
            resp = frame["response"]
            assert resp["request_id"] == "ctrl-q1"
            assert resp["response"]["behavior"] == "allow"
            transport.push(
                {
                    "type": "assistant",
                    "session_id": "claude-sess-42",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tu-1",
                                "name": "Edit",
                                "input": {"file_path": "main.py"},
                            }
                        ]
                    },
                }
            )
            transport.push(
                {
                    "type": "user",
                    "session_id": "claude-sess-42",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tu-1",
                                "content": "Updated main.py",
                            }
                        ]
                    },
                }
            )
            transport.push(
                {
                    "type": "stream_event",
                    "event": {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": "Edited main.py."},
                    },
                }
            )
            transport.push(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": "Edited main.py.",
                    "session_id": "claude-sess-42",
                }
            )

    transport = FakeStdioTransport(handle_write)

    async def factory(cmd, cwd, env):
        captured_cmd.extend(cmd)
        return transport

    session = ClaudeHarnessSession(
        SessionConfig(
            thread_id=uuid4(),
            working_dir=str(tmp_path),
            model="claude-sonnet-4-6",
            harness_session_id="prev-session",
        ),
        binary="claude",
        credential="sk-ant-oat01-test",
        process_factory=factory,
    )

    events = []
    async for ev in session.start_turn("Update main.py"):
        events.append(ev)
        if ev.kind == "pending_question":
            await session.respond_pending(ev.request_id, {"answer": "main"})

    await session.close()
    assert "--resume" in captured_cmd
    assert "prev-session" in captured_cmd
    assert session.session_id == "claude-sess-42"
    assert any(
        e.kind == "pending_question" and "Which branch" in e.text for e in events
    )
    assert any(e.kind == "file_change" and e.file_path == "main.py" for e in events)
    assert any(e.kind == "tool_result" and "Updated main.py" in e.text for e in events)
    assert events[-1].kind == "turn_completed"
    assert events[-1].terminal_status == "completed"


@pytest.mark.anyio
async def test_acp_v2_session_permission_roundtrip_and_updates(tmp_path):
    init_versions: list[Any] = []

    def handle_write(frame: dict[str, Any], transport: FakeStdioTransport) -> None:
        method = frame.get("method")
        req_id = frame.get("id")
        if method == "initialize":
            init_versions.append(frame["params"]["protocolVersion"])
            transport.push(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": 2,
                        "agentInfo": {"name": "antigravity"},
                    },
                }
            )
        elif method == "session/new":
            transport.push(
                {"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": "acp-v2-1"}}
            )
        elif method == "session/set_model":
            transport.push({"jsonrpc": "2.0", "id": req_id, "result": {}})
        elif method == "session/prompt":
            transport._prompt_req_id = req_id
            transport.push(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": "acp-v2-1",
                        "update": {
                            "sessionUpdate": "tool_call",
                            "toolCallId": "tc-1",
                            "title": "Run build",
                        },
                    },
                }
            )
            transport.push(
                {
                    "jsonrpc": "2.0",
                    "id": "perm-44",
                    "method": "session/request_permission",
                    "params": {
                        "toolCall": {"title": "Run build"},
                        "options": [
                            {
                                "optionId": "allow_once",
                                "name": "Allow once",
                                "kind": "allow_once",
                            },
                            {
                                "optionId": "reject_once",
                                "name": "Reject",
                                "kind": "reject_once",
                            },
                        ],
                    },
                }
            )
        elif "result" in frame and frame.get("id") == "perm-44":
            assert frame["result"]["outcome"]["optionId"] == "allow_once"
            transport.push(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": "acp-v2-1",
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "Build succeeded."},
                        },
                    },
                }
            )
            transport.push(
                {
                    "jsonrpc": "2.0",
                    "id": transport._prompt_req_id,
                    "result": {"stopReason": "end_turn"},
                }
            )

    transport = FakeStdioTransport(handle_write)

    async def factory(cmd, cwd, env):
        return transport

    class DummyServices:
        async def json_request(self, *args, **kwargs):
            return None

    adapter = AntigravityHarnessAdapter(process_factory=factory)
    session = await adapter.open_session(
        SessionConfig(
            thread_id=uuid4(), working_dir=str(tmp_path), model="gemini-2.5-pro"
        ),
        DummyServices(),  # type: ignore[arg-type]
    )
    assert isinstance(session, AcpHarnessSession)

    events = []
    async for ev in session.start_turn("Build project"):
        events.append(ev)
        if ev.kind == "pending_approval":
            await session.respond_pending(ev.request_id, {"decision": "approve"})

    await session.close()
    assert init_versions == [ACP_V2_VERSION]
    assert session.protocol_version == ACP_V2_VERSION
    assert session.session_id == "acp-v2-1"
    assert any(e.kind == "pending_approval" for e in events)
    assert events[-1].kind == "turn_completed"
    assert events[-1].text == "Build succeeded."


@pytest.mark.anyio
async def test_acp_falls_back_to_v1_and_loads_session_for_grok(tmp_path):
    init_versions: list[Any] = []
    seen_methods: list[str] = []

    def handle_write(frame: dict[str, Any], transport: FakeStdioTransport) -> None:
        method = frame.get("method")
        req_id = frame.get("id")
        if method:
            seen_methods.append(method)
        if method == "initialize":
            ver = frame["params"]["protocolVersion"]
            init_versions.append(ver)
            if ver == ACP_V2_VERSION:
                transport.push(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32602,
                            "message": "Unsupported protocol version 2",
                        },
                    }
                )
            else:
                transport.push(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"protocolVersion": 1},
                    }
                )
        elif method == "session/load":
            transport.push(
                {"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": "grok-v1-7"}}
            )
        elif method == "session/set_model":
            transport.push({"jsonrpc": "2.0", "id": req_id, "result": {}})
        elif method == "session/prompt":
            transport.push(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "Grok v1 reply"},
                        }
                    },
                }
            )
            transport.push(
                {"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn"}}
            )

    transport = FakeStdioTransport(handle_write)

    async def factory(cmd, cwd, env):
        return transport

    class DummyServices:
        async def json_request(self, *args, **kwargs):
            return {"credential": "xai-key-123", "account_label": "xAI"}

    adapter = GrokHarnessAdapter(process_factory=factory)
    session = await adapter.open_session(
        SessionConfig(
            thread_id=uuid4(),
            working_dir=str(tmp_path),
            model="grok-code-fast-1",
            harness_session_id="grok-v1-7",
        ),
        DummyServices(),  # type: ignore[arg-type]
    )
    assert isinstance(session, AcpHarnessSession)

    events = [ev async for ev in session.start_turn("Explain bug")]
    await session.close()

    assert init_versions == [ACP_V2_VERSION, ACP_V1_VERSION]
    assert session.protocol_version == ACP_V1_VERSION
    assert "session/load" in seen_methods
    assert events[-1].kind == "turn_completed"
    assert events[-1].text == "Grok v1 reply"


@pytest.mark.anyio
async def test_claude_session_forwards_effort_flag(tmp_path):
    captured_cmd: list[str] = []

    def handle_write(frame: dict[str, Any], transport: FakeStdioTransport) -> None:
        if frame.get("type") == "user":
            transport.push(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": "Done",
                    "session_id": "claude-eff-1",
                }
            )

    transport = FakeStdioTransport(handle_write)

    async def factory(cmd, cwd, env):
        captured_cmd.extend(cmd)
        return transport

    session = ClaudeHarnessSession(
        SessionConfig(
            thread_id=uuid4(),
            working_dir=str(tmp_path),
            model="claude-opus-5-5",
            model_options={"effort": "xhigh"},
        ),
        binary="claude",
        process_factory=factory,
    )
    events = [ev async for ev in session.start_turn("Hello")]
    await session.close()
    assert "--model" in captured_cmd
    assert captured_cmd[captured_cmd.index("--model") + 1] == "claude-opus-5-5"
    assert "--effort" in captured_cmd
    assert captured_cmd[captured_cmd.index("--effort") + 1] == "xhigh"
    assert events[-1].terminal_status == "completed"


@pytest.mark.anyio
async def test_codex_adapter_discovers_paginated_models_and_caches(monkeypatch):
    spawn_count = 0

    def handle_write(frame: dict[str, Any], transport: FakeStdioTransport) -> None:
        method = frame.get("method")
        req_id = frame.get("id")
        if method == "initialize":
            transport.push({"id": req_id, "result": {"userAgent": "codex"}})
        elif method == "model/list":
            cursor = (frame.get("params") or {}).get("cursor")
            if not cursor:
                transport.push(
                    {
                        "id": req_id,
                        "result": {
                            "data": [
                                {
                                    "id": "gpt-6.2-sol",
                                    "displayName": "GPT-6.2-Sol",
                                    "description": "Default model",
                                    "hidden": False,
                                    "isDefault": True,
                                    "supportedReasoningEfforts": [
                                        {"reasoningEffort": "low"},
                                        {"reasoningEffort": "high"},
                                        {"reasoningEffort": "ultra"},
                                    ],
                                },
                                {
                                    "id": "hidden-internal",
                                    "displayName": "Hidden",
                                    "hidden": True,
                                },
                            ],
                            "nextCursor": "page-2",
                        },
                    }
                )
            else:
                transport.push(
                    {
                        "id": req_id,
                        "result": {
                            "data": [
                                {
                                    "id": "gpt-6.2-astra",
                                    "displayName": "GPT-6.2-Astra",
                                    "description": "Most capable",
                                    "hidden": False,
                                    "isDefault": False,
                                    "supportedReasoningEfforts": [
                                        {"reasoningEffort": "medium"},
                                        {"reasoningEffort": "max"},
                                    ],
                                }
                            ],
                            "nextCursor": None,
                        },
                    }
                )

    async def factory(cmd, cwd, env):
        nonlocal spawn_count
        spawn_count += 1
        return FakeStdioTransport(handle_write)

    class DummyServices:
        async def json_request(self, service, method, path, **kwargs):
            if method == "GET":
                return {
                    "credential": '{"access_token":"tok"}',
                    "account_label": "Codex",
                }
            return {}

    monkeypatch.setattr(
        "kadan.work.codex_adapter._find_codex_binary", lambda: "/usr/bin/codex"
    )
    adapter = CodexHarnessAdapter(process_factory=factory)
    first = await adapter.check_availability(DummyServices())  # type: ignore[arg-type]
    second = await adapter.check_availability(DummyServices())  # type: ignore[arg-type]

    assert spawn_count == 1
    assert first.default_model == "gpt-6.2-sol"
    assert [m.id for m in first.models] == ["gpt-6.2-sol", "gpt-6.2-astra"]
    assert first.models[0].options == {"effort": ["low", "high", "ultra"]}
    assert second.models == first.models

    # Signing out invalidates the model cache.
    await adapter.sign_out(DummyServices())  # type: ignore[arg-type]
    await adapter.check_availability(DummyServices())  # type: ignore[arg-type]
    assert spawn_count == 2


@pytest.mark.anyio
async def test_claude_adapter_discovers_models_via_initialize_and_dedupes_default(
    monkeypatch,
):
    spawn_count = 0

    def handle_write(frame: dict[str, Any], transport: FakeStdioTransport) -> None:
        if (
            frame.get("type") == "control_request"
            and (frame.get("request") or {}).get("subtype") == "initialize"
        ):
            transport.push(
                {
                    "type": "control_response",
                    "response": {
                        "subtype": "success",
                        "request_id": frame["request_id"],
                        "response": {
                            "models": [
                                {
                                    "value": "default",
                                    "resolvedModel": "claude-opus-5-5",
                                    "displayName": "Default (recommended)",
                                    "description": "Opus 5.5",
                                    "supportedEffortLevels": ["low", "high", "max"],
                                },
                                {
                                    "value": "opus",
                                    "resolvedModel": "claude-opus-5-5",
                                    "displayName": "Opus 5.5",
                                    "description": "Complex work",
                                    "supportedEffortLevels": ["low", "high", "max"],
                                },
                                {
                                    "value": "sonnet",
                                    "resolvedModel": "claude-sonnet-5-5",
                                    "displayName": "Sonnet 5.5",
                                    "description": "Efficient",
                                    "supportedEffortLevels": ["low", "medium"],
                                },
                                {
                                    "value": "claude-haiku-4-5-20251001",
                                    "resolvedModel": "claude-haiku-4-5-20251001",
                                    "displayName": "Haiku 4.5",
                                    "description": "Fast",
                                },
                            ]
                        },
                    },
                }
            )

    async def factory(cmd, cwd, env):
        nonlocal spawn_count
        spawn_count += 1
        return FakeStdioTransport(handle_write)

    class DummyServices:
        async def json_request(self, service, method, path, **kwargs):
            if method == "GET":
                return {"credential": "sk-ant-oat01-test", "account_label": "Claude"}
            return {}

    monkeypatch.setattr(
        "kadan.work.claude_adapter._find_claude_binary", lambda: "/usr/bin/claude"
    )
    adapter = ClaudeHarnessAdapter(process_factory=factory)
    avail = await adapter.check_availability(DummyServices())  # type: ignore[arg-type]

    assert spawn_count == 1
    assert avail.default_model == "claude-opus-5-5"
    assert [m.id for m in avail.models] == [
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-haiku-4-5-20251001",
    ]
    assert [m.label for m in avail.models] == ["Opus 5.5", "Sonnet 5.5", "Haiku 4.5"]
    assert avail.models[0].options == {"effort": ["low", "high", "max"]}
    assert avail.models[2].options == {}


@pytest.mark.anyio
async def test_antigravity_adapter_discovers_acp_models_and_config_options(monkeypatch):
    def handle_write(frame: dict[str, Any], transport: FakeStdioTransport) -> None:
        method = frame.get("method")
        req_id = frame.get("id")
        if method == "initialize":
            transport.push(
                {"jsonrpc": "2.0", "id": req_id, "result": {"protocolVersion": 2}}
            )
        elif method == "session/new":
            transport.push(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "sessionId": "probe-1",
                        "configOptions": [
                            {
                                "id": "model",
                                "category": "model",
                                "currentValue": "gemini-4-neon",
                                "options": [
                                    {
                                        "group": "Gemini",
                                        "options": [
                                            {
                                                "value": "gemini-4-neon",
                                                "name": "Gemini 4 Neon",
                                                "description": "Flagship model",
                                            },
                                            {
                                                "value": "gemini-4-flash",
                                                "name": "Gemini 4 Flash",
                                                "description": "Fast model",
                                            },
                                        ],
                                    }
                                ],
                            },
                            {
                                "id": "thought_level",
                                "category": "thought_level",
                                "currentValue": "medium",
                                "options": [
                                    {"value": "low", "name": "Low"},
                                    {"value": "medium", "name": "Medium"},
                                    {"value": "high", "name": "High"},
                                ],
                            },
                        ],
                    },
                }
            )
        elif method == "session/close":
            transport.push({"jsonrpc": "2.0", "id": req_id, "result": {}})

    async def factory(cmd, cwd, env):
        return FakeStdioTransport(handle_write)

    class DummyServices:
        async def json_request(self, service, method, path, **kwargs):
            return {"credential": "google-key", "account_label": "Google"}

    monkeypatch.setattr(
        "kadan.work.antigravity_adapter.find_antigravity_command",
        lambda: ["/usr/bin/agy", "acp"],
    )
    adapter = AntigravityHarnessAdapter(process_factory=factory)
    avail = await adapter.check_availability(DummyServices())  # type: ignore[arg-type]

    assert avail.default_model == "gemini-4-neon"
    assert [m.id for m in avail.models] == ["gemini-4-neon", "gemini-4-flash"]
    assert avail.models[0].label == "Gemini 4 Neon"
    assert avail.models[0].options == {"effort": ["low", "medium", "high"]}


def test_model_parsers_for_acp_grok_and_agy():
    acp_models, acp_default = parse_acp_models(
        {},
        {
            "models": {
                "currentModelId": "grok-4.6",
                "availableModels": [
                    {"modelId": "grok-4.6", "name": "Grok 4.6"},
                    {"modelId": "grok-4.5", "name": "Grok 4.5"},
                ],
            }
        },
    )
    assert acp_default == "grok-4.6"
    assert [m.id for m in acp_models] == ["grok-4.6", "grok-4.5"]

    grok_models, grok_default = _parse_grok_models_output(
        "You are not authenticated.\n\nDefault model: grok-4.6\n\n"
        "Available models:\n  * grok-4.6 (default)\n  - grok-4.5\n"
    )
    assert grok_default == "grok-4.6"
    assert [(m.id, m.label) for m in grok_models] == [
        ("grok-4.6", "Grok 4.6"),
        ("grok-4.5", "Grok 4.5"),
    ]

    agy_models, agy_default = _parse_agy_models_output(
        "Available models:\n  * gemini-4-neon (default)\n  - gemini-4-flash\n"
    )
    assert agy_default == "gemini-4-neon"
    assert [m.id for m in agy_models] == ["gemini-4-neon", "gemini-4-flash"]

    codex_models, codex_default = _parse_codex_models([])
    assert codex_models == [] and codex_default is None
    claude_models, claude_default = _parse_claude_models([])
    assert claude_models == [] and claude_default is None


@pytest.mark.anyio
async def test_model_catalog_cache_coalesces_and_reloads_on_key_change():
    cache = ModelCatalogCache(ttl_seconds=60.0, error_ttl_seconds=10.0)
    calls = 0

    async def loader():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return [HarnessModelInfo(id=f"m-{calls}", label=f"Model {calls}")], f"m-{calls}"

    r1, r2 = await asyncio.gather(
        cache.get_or_load("k1", loader),
        cache.get_or_load("k1", loader),
    )
    assert calls == 1
    assert r1 == r2

    r3 = await cache.get_or_load("k2", loader)
    assert calls == 2
    assert r3[1] == "m-2"

    cache.invalidate()
    r4 = await cache.get_or_load("k2", loader)
    assert calls == 3
    assert r4[1] == "m-3"
