import asyncio
import json
from contextlib import asynccontextmanager
from typing import ClassVar

import httpx
import pytest
from mcp.types import CallToolResult, TextContent, Tool

from kadan.api.config import ServiceError, Settings
from kadan.api.pydantic_models.plugins import PluginSkill
from kadan.api.pydantic_models.resources import Message
from kadan.api.routes.agents.conversations.chat import ChatInput
from kadan.api.services import LocalServices, completion_messages
from kadan.api.tool_loop import ToolExecution

from .test_services import CONVERSATION_ID, MESSAGE_ID, Boundary, generate, resource


class Session:
    tools: ClassVar = [
        Tool(
            name="fixture__remember",
            description="Save memory",
            inputSchema={"type": "object"},
        )
    ]
    skills: ClassVar = []

    def __init__(self, result=None):
        self.calls = []
        self.result = result or CallToolResult(
            content=[TextContent(type="text", text="Saved")]
        )
        self.closed = False

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self.result


class Plugins:
    def __init__(self, session):
        self.session = session

    def select(self, ids):
        if ids == ["missing"]:
            raise ServiceError("Unknown plugin", 422, "unknown_plugin")
        return ["fixture"]

    @asynccontextmanager
    async def connect(self, services, key, ids):
        try:
            yield self.session
        finally:
            self.session.closed = True


class ToolBoundary(Boundary):
    def __init__(self, decisions):
        super().__init__()
        self.decisions = iter(decisions)
        self.plans = []

    def __call__(self, request):
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content)
            if not body["stream"]:
                self.plans.append(body)
                decision = next(self.decisions)
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {"message": {"content": decision}, "finish_reason": "stop"}
                        ]
                    },
                )
        return super().__call__(request)


def run_tool_chat(boundary, session=None):
    session = session or Session()

    async def operation():
        async with httpx.AsyncClient(transport=httpx.MockTransport(boundary)) as client:
            services = LocalServices(client, Settings(_env_file=None, plugins=[]))
            services.plugins = Plugins(session)
            return await generate(services)

    return asyncio.run(operation()), session


CALL = '{"type":"tool","name":"fixture__remember","arguments":{"title":"Note"}}'
REPLY = '{"type":"reply"}'


def test_tool_intent_and_result_are_saved_before_reply_and_replayed_to_model():
    boundary = ToolBoundary([CALL, REPLY])
    frames, session = run_tool_chat(boundary)
    assert session.calls == [("fixture__remember", {"title": "Note"})]
    assert session.closed
    assert [message.role for message in boundary.current.messages] == [
        "user",
        "tool",
        "tool",
        "assistant",
    ]
    start, result = [
        ToolExecution.model_validate_json(m.content)
        for m in boundary.current.messages[1:3]
    ]
    assert start.id == result.id
    assert start.status == "started" and result.status == "completed"
    assert [entry["expected_version"] for entry in boundary.appends] == [1, 2, 3, 4]
    assert [name for name, _ in frames].count("tool_started") == 1
    assert frames[-1][0] == "done"
    assert "Saved" in boundary.plans[-1]["messages"][-1]["content"]
    assert "Saved" in boundary.inference_body["messages"][-1]["content"]


def test_tool_error_is_visible_saved_and_available_for_model_recovery():
    session = Session(
        CallToolResult(
            isError=True, content=[TextContent(type="text", text="Version conflict")]
        )
    )
    boundary = ToolBoundary([CALL, REPLY])
    frames, _ = run_tool_chat(boundary, session)
    outcome = next(
        data["execution"] for name, data in frames if name == "tool_finished"
    )
    assert outcome["status"] == "error"
    assert "Version conflict" in boundary.plans[-1]["messages"][-1]["content"]
    assert frames[-1][0] == "done"


@pytest.mark.parametrize(
    "decision",
    [
        "not json",
        '{"type":"tool","name":"fixture__remember"}',
        REPLY + " trailing",
        '{"type":"reply","extra":true}',
    ],
)
def test_invalid_decisions_cannot_execute_tools(decision):
    boundary = ToolBoundary([decision])
    frames, session = run_tool_chat(boundary)
    assert not session.calls
    assert len(boundary.current.messages) == 1
    assert frames[-1][1]["code"] == "invalid_tool_action"


def test_tool_budget_terminates_repeated_model_actions():
    boundary = ToolBoundary([CALL] * 9)
    frames, session = run_tool_chat(boundary)
    assert len(session.calls) == 8
    assert frames[-1][1]["code"] == "tool_limit"
    assert not any(m.role == "assistant" for m in boundary.current.messages)


def test_stale_intent_save_prevents_tool_side_effect():
    class ConflictBoundary(ToolBoundary):
        def __call__(self, request):
            if (
                request.url.path.endswith("/messages")
                and json.loads(request.content)["message"]["role"] == "tool"
            ):
                return httpx.Response(409, json={"detail": "Concurrent edit"})
            return super().__call__(request)

    frames, session = run_tool_chat(ConflictBoundary([CALL]))
    assert not session.calls
    assert frames[-1][1]["code"] == "version_conflict"


def test_stale_result_save_reports_unknown_outcome_without_repeating_action():
    class ConflictBoundary(ToolBoundary):
        def __call__(self, request):
            if request.url.path.endswith("/messages"):
                message = json.loads(request.content)["message"]
                if (
                    message["role"] == "tool"
                    and json.loads(message["content"])["status"] == "completed"
                ):
                    return httpx.Response(409, json={"detail": "Concurrent edit"})
            return super().__call__(request)

    boundary = ConflictBoundary([CALL])
    frames, session = run_tool_chat(boundary)
    assert len(session.calls) == 1
    assert frames[-1][1]["code"] == "version_conflict"
    assert (
        ToolExecution.model_validate_json(boundary.current.messages[-1].content).status
        == "started"
    )


def test_tool_history_remains_data_and_documents_unknown_outcomes():
    messages = completion_messages(
        resource(messages=[Message(role="tool", content='{"status":"started"}')]),
        None,
        Message(role="user", content="Next"),
    )
    assert messages[0].role == "user"
    assert "unknown outcome" in messages[0].content


def test_unknown_plugin_rejected_before_user_is_saved():
    boundary = Boundary()

    async def operation():
        async with httpx.AsyncClient(transport=httpx.MockTransport(boundary)) as client:
            services = LocalServices(client, Settings(_env_file=None, plugins=[]))
            services.plugins = Plugins(Session())
            with pytest.raises(ServiceError, match="Unknown plugin"):
                await services.prepare_chat(
                    "test",
                    CONVERSATION_ID,
                    ChatInput(
                        content="Hi",
                        expected_version=1,
                        message_id=MESSAGE_ID,
                        persona_id=None,
                        plugin_ids=["missing"],
                    ),
                )

    asyncio.run(operation())
    assert boundary.appends == []


def test_cancellation_during_call_preserves_intent_and_closes_session():
    class BlockingSession(Session):
        async def call_tool(self, name, arguments):
            self.calls.append((name, arguments))
            started.set()
            await asyncio.Event().wait()

    async def operation():
        nonlocal started
        started = asyncio.Event()
        boundary = ToolBoundary([CALL])
        session = BlockingSession()
        async with httpx.AsyncClient(transport=httpx.MockTransport(boundary)) as client:
            service = LocalServices(client, Settings(_env_file=None, plugins=[]))
            service.plugins = Plugins(session)
            task = asyncio.create_task(generate(service))
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert session.closed
        assert len(session.calls) == 1
        assert (
            ToolExecution.model_validate_json(
                boundary.current.messages[-1].content
            ).status
            == "started"
        )

    started = None
    asyncio.run(operation())


@pytest.mark.parametrize(
    "decision",
    ["<think>reasoning</think>" + REPLY, "<think>reasoning</think>\n" + CALL],
)
def test_complete_hikari_reasoning_envelope_uses_only_final_json(decision):
    boundary = ToolBoundary([decision, REPLY])
    frames, _ = run_tool_chat(boundary)
    assert frames[-1][0] == "done"


@pytest.mark.parametrize(
    "decision",
    [
        "<think>" + CALL,
        "<think>open<think>nested</think>" + CALL,
        "<think>x</think>" + CALL + " garbage",
    ],
)
def test_incomplete_or_ambiguous_reasoning_cannot_dispatch_tool(decision):
    frames, session = run_tool_chat(ToolBoundary([decision]))
    assert not session.calls
    assert frames[-1][1]["code"] == "invalid_tool_action"


def test_persona_skills_and_planner_use_one_leading_system_message():
    session = Session()
    session.skills = [
        PluginSkill(
            name="Fixture",
            description="Fixture skill",
            instructions="Read before editing.",
        )
    ]
    boundary = ToolBoundary([REPLY])
    frames, _ = run_tool_chat(boundary, session)
    for body in [boundary.plans[0], boundary.inference_body]:
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
        assert "Read before editing." in body["messages"][0]["content"]
        assert "Be precise." in body["messages"][0]["content"]
    assert frames[-1][0] == "done"


def test_terminal_tool_timeout_closes_session_and_never_plans_again():
    class TimedOutSession(Session):
        async def call_tool(self, name, arguments):
            self.calls.append((name, arguments))
            raise ServiceError(
                "Tool timed out; write outcome unknown.", 504, "tool_timeout"
            )

    boundary = ToolBoundary([CALL])
    frames, session = run_tool_chat(boundary, TimedOutSession())
    assert session.closed
    assert len(session.calls) == len(boundary.plans) == 1
    assert frames[-1][1]["code"] == "tool_timeout"
    assert (
        ToolExecution.model_validate_json(boundary.current.messages[-1].content).status
        == "started"
    )
    assert not any(message.role == "assistant" for message in boundary.current.messages)
