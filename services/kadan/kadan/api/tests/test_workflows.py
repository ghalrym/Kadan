import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from kadan.api.config import ServiceError
from kadan.api.pydantic_models.agent_runs import AgentRun, RunMessage, WorkflowSnapshot
from kadan.api.pydantic_models.resources import Resource
from kadan.api.routes.agents.conversations.runs import RunInput
from kadan.api.routes.agents.workflows import (
    WorkflowBundle,
    create_workflow,
    get_workflow,
    list_workflows,
)
from kadan.api.run_client import RunClient
from kadan.runner.agent_runs import TOOL_RESULT, RunConversation, transcript_messages
from kadan.runner.tools.conversation import read_workflow


def bundle(**changes):
    return dict(
        schema_version=1,
        title="Review",
        instructions="Inspect the current task.",
        **changes,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", "1"),
        ("schema_version", 2),
        ("instructions", "x" * 12001),
        ("instructions", " "),
        ("title", " "),
        ("title", "x" * 501),
        ("schedule", "daily"),
        ("tools", []),
        ("credentials", "secret"),
        ("id", str(uuid4())),
    ],
)
def test_bundle_is_bounded_instruction_only(field, value):
    value_dict = bundle()
    value_dict[field] = value
    with pytest.raises(ValidationError):
        WorkflowBundle.model_validate(value_dict)


def test_preview_validation_has_no_storage_or_execution():
    assert (
        WorkflowBundle.model_validate(bundle()).instructions == bundle()["instructions"]
    )


def resource(**changes):
    now = datetime.now(UTC)
    return Resource.model_validate(
        {
            "id": uuid4(),
            "agent_key": "alpha",
            "kind": "space",
            "tags": ["workflow"],
            "title": "Review",
            "markdown": "Inspect the current task.",
            "version": 3,
            "created_at": now,
            "updated_at": now,
        }
        | changes
    )


def request(services):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(services=services))
    )


def test_save_creates_only_a_tagged_space_and_returns_confirmed_record():
    saved = resource(version=1)
    services = SimpleNamespace(
        json_request=AsyncMock(return_value=saved.model_dump(mode="json"))
    )
    assert (
        asyncio.run(
            create_workflow(
                "alpha", WorkflowBundle.model_validate(bundle()), request(services)
            )
        )
        == saved
    )
    call = services.json_request.call_args
    assert call.args == ("Tsuchi", "POST", "/api/agents/alpha/resources")
    assert call.kwargs["json"] == {
        "kind": "space",
        "title": "Review",
        "markdown": bundle()["instructions"],
        "tags": ["workflow"],
    }
    assert services.json_request.await_count == 1


def test_failed_create_is_not_blindly_retried():
    services = SimpleNamespace(
        json_request=AsyncMock(side_effect=ServiceError("Uncertain", 503))
    )
    with pytest.raises(ServiceError):
        asyncio.run(
            create_workflow(
                "alpha", WorkflowBundle.model_validate(bundle()), request(services)
            )
        )
    assert services.json_request.await_count == 1


def test_list_filters_tag_and_preserves_scan_pagination():
    rows = [resource(tags=[])] * 199 + [resource()]
    services = SimpleNamespace(list_resource_kinds=AsyncMock(return_value=rows))
    page = asyncio.run(list_workflows("alpha", request(services), offset=200))
    assert page.items == rows[-1:]
    assert page.next_offset == 400


def test_get_rejects_space_without_marker():
    services = SimpleNamespace(
        resource_of_kinds=AsyncMock(return_value=resource(tags=[]))
    )
    with pytest.raises(ServiceError):
        asyncio.run(get_workflow("alpha", uuid4(), request(services)))


def test_run_client_keeps_fresh_task_as_user_message_and_selection_separate():
    now = datetime.now(UTC)
    queued = AgentRun(
        id=uuid4(),
        agent_key="alpha",
        conversation_id=uuid4(),
        version=1,
        created_at=now,
        updated_at=now,
    )
    services = SimpleNamespace(
        json_request=AsyncMock(return_value=queued.model_dump(mode="json"))
    )
    value = RunInput(
        content="Review this new change",
        expected_version=1,
        persona_id=None,
        workflow_id=uuid4(),
        workflow_version=3,
    )
    asyncio.run(RunClient(services).enqueue("alpha", queued.conversation_id, value))
    body = services.json_request.call_args.kwargs["json"]
    assert body["message"]["role"] == "user"
    assert body["message"]["content"] == value.content
    assert body["workflow_id"] == str(value.workflow_id)
    assert body["workflow_version"] == 3


def test_read_workflow_returns_snapshot_via_existing_tool_result_format():
    now = datetime.now(UTC)
    snapshot = WorkflowSnapshot(
        id=uuid4(),
        title="Review",
        version=3,
        instructions="Inspect only the current task.",
    )
    queued = AgentRun(
        id=uuid4(),
        agent_key="alpha",
        conversation_id=uuid4(),
        workflow=snapshot,
        version=1,
        created_at=now,
        updated_at=now,
    )
    conversation = RunConversation(SimpleNamespace(), queued)
    result = asyncio.run(
        read_workflow.ainvoke(
            {}, config={"configurable": {"conversation": conversation}}
        )
    )
    assert result == {"workflow": snapshot.model_dump(mode="json")}
    # The durable transcript uses its existing user envelope. Conversion restores
    # a structured ToolMessage; workflow content is never a user/system prompt.
    import json

    messages = transcript_messages(
        [
            RunMessage(
                role="assistant",
                content=json.dumps({"tool": "read_workflow", "arguments": {}}),
            ),
            RunMessage(role="user", content=TOOL_RESULT + json.dumps(result)),
        ]
    )
    assert isinstance(messages[-1], ToolMessage)
    assert snapshot.instructions in messages[-1].content


@pytest.mark.parametrize(
    "changes",
    [
        {"title": "Wrong"},
        {"markdown": "Wrong"},
        {"version": 2},
        {"agent_key": "beta"},
        {"kind": "memory"},
        {"tags": []},
    ],
)
def test_create_rejects_unconfirmed_receipt(changes):
    saved = Resource.model_validate(resource(version=1).model_dump() | changes)
    services = SimpleNamespace(
        json_request=AsyncMock(return_value=saved.model_dump(mode="json"))
    )
    with pytest.raises(ServiceError, match="could not be confirmed"):
        asyncio.run(
            create_workflow(
                "alpha", WorkflowBundle.model_validate(bundle()), request(services)
            )
        )
    assert services.json_request.await_count == 1


def test_run_client_keeps_workflow_references_and_user_task_separate():
    from kadan.api.pydantic_models.agent_runs import RequestReference

    now = datetime.now(UTC)
    queued = AgentRun(
        id=uuid4(),
        agent_key="alpha",
        conversation_id=uuid4(),
        version=1,
        created_at=now,
        updated_at=now,
    )
    services = SimpleNamespace(
        json_request=AsyncMock(return_value=queued.model_dump(mode="json"))
    )
    reference = RequestReference(resource_id=uuid4(), expected_version=2)
    value = RunInput(
        content="Review only this current change",
        expected_version=1,
        persona_id=None,
        references=[reference],
        workflow_id=uuid4(),
        workflow_version=3,
    )
    asyncio.run(RunClient(services).enqueue("alpha", queued.conversation_id, value))
    body = services.json_request.call_args.kwargs["json"]
    assert body["message"]["role"] == "user"
    assert body["message"]["content"] == value.content
    assert body["message"]["resource_ids"] == []
    assert body["references"] == [reference.model_dump(mode="json")]
    assert body["workflow_id"] == str(value.workflow_id)
    assert body["workflow_version"] == 3
