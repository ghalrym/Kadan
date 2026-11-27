"""Unit tests for Codex app-server, Claude Code stream-json, and ACP v2/v1 adapters with mocked stdio."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any
from uuid import uuid4

import pytest

from kadan.work.acp_client import ACP_V1_VERSION, ACP_V2_VERSION, AcpHarnessSession
from kadan.work.antigravity_adapter import AntigravityHarnessAdapter
from kadan.work.claude_adapter import ClaudeHarnessSession
from kadan.work.codex_adapter import CodexHarnessSession
from kadan.work.contract import SessionConfig
from kadan.work.grok_adapter import GrokHarnessAdapter


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
