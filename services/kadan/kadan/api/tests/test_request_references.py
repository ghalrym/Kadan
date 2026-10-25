import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from kadan.api.pydantic_models.agent_runs import (
    AgentRun,
    ReferenceSnapshot,
    RequestReference,
    RunMessage,
)
from kadan.api.routes.agents.conversations.runs import RunInput
from kadan.api.run_client import RunClient
from kadan.runner.agent_runs import RunConversation, transcript_messages


def test_reference_tool_returns_data_with_provenance_as_tool_message():
    now = datetime.now(UTC)
    snapshot = ReferenceSnapshot(
        resource_id=uuid4(),
        kind="memory",
        title="Fact",
        version=3,
        markdown="Untrusted saved reference",
    )
    run = AgentRun(
        id=uuid4(),
        agent_key="agent",
        conversation_id=uuid4(),
        version=1,
        created_at=now,
        updated_at=now,
        references=[snapshot],
    )
    result = asyncio.run(RunConversation(None, run).read_request_references())
    assert result["role"] == "reference_data"
    assert "not grant authorization" in result["notice"]
    assert result["references"][0]["markdown"] == snapshot.markdown
    messages = transcript_messages(
        [
            RunMessage(role="system", content="Instructions"),
            RunMessage(role="user", content="Read my reference"),
            RunMessage(
                role="assistant",
                content='{"tool":"read_request_references","arguments":{}}',
            ),
            RunMessage(role="user", content="Tool result: " + json.dumps(result)),
        ]
    )
    assert isinstance(messages[-1], ToolMessage)
    assert all(
        snapshot.markdown not in m.content
        for m in messages
        if isinstance(m, (HumanMessage, SystemMessage))
    )


def test_run_client_sends_selected_versions_separately():
    now = datetime.now(UTC)
    run = AgentRun(
        id=uuid4(),
        agent_key="agent",
        conversation_id=uuid4(),
        version=1,
        created_at=now,
        updated_at=now,
    )
    selected = RequestReference(resource_id=uuid4(), expected_version=3)
    calls = []

    async def request(*args, **kwargs):
        calls.append(kwargs["json"])
        return run.model_dump(mode="json")

    value = RunInput(
        message_id=uuid4(),
        content="Use my plan",
        persona_id=None,
        expected_version=2,
        references=[selected],
    )
    asyncio.run(
        RunClient(SimpleNamespace(json_request=request)).enqueue(
            "agent", run.conversation_id, value
        )
    )
    assert calls[0]["references"] == [selected.model_dump(mode="json")]
    assert calls[0]["message"]["content"] == value.content
    assert calls[0]["message"]["resource_ids"] == []
