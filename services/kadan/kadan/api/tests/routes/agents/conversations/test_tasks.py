import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from kadan.api.config import ServiceError
from kadan.api.pydantic_models.tasks import AssistantTask, TaskControl
from kadan.api.routes.agents.conversations.tasks import (
    control_task,
    conversation_tasks,
)


@pytest.fixture
def boundary():
    now = datetime.now(UTC)
    task = AssistantTask(
        id=uuid4(),
        agent_key="alpha",
        conversation_id=uuid4(),
        origin_run_id=uuid4(),
        status="waiting",
        goal="Prepare the proposal",
        progress="Outline saved",
        next_step="Ask about the budget",
        blocker="Budget needed",
        wait_kind="user",
        version=3,
        created_at=now,
        updated_at=now,
    )
    services = SimpleNamespace(
        resource_of_kinds=AsyncMock(),
        json_request=AsyncMock(return_value=[task.model_dump(mode="json")]),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(services=services))
    )
    return task, services, request


def test_lists_only_the_requested_conversations_saved_goals(boundary):
    task, services, request = boundary
    result = asyncio.run(conversation_tasks("alpha", task.conversation_id, request))
    assert result == [task]
    services.resource_of_kinds.assert_awaited_once()
    services.json_request.assert_awaited_once_with(
        "Tsuchi", "GET", f"/api/agents/alpha/resources/{task.conversation_id}/tasks"
    )


@pytest.mark.parametrize(
    "field,value", [("agent_key", "beta"), ("conversation_id", uuid4())]
)
def test_rejects_goals_from_another_scope(boundary, field, value):
    task, services, request = boundary
    services.json_request.return_value = [
        {**task.model_dump(mode="json"), field: str(value)}
    ]
    with pytest.raises(ServiceError):
        asyncio.run(conversation_tasks("alpha", task.conversation_id, request))


def test_upstream_failure_remains_failure_instead_of_empty_goals(boundary):
    task, services, request = boundary
    services.json_request.side_effect = ServiceError("Unavailable", 503)
    with pytest.raises(ServiceError):
        asyncio.run(conversation_tasks("alpha", task.conversation_id, request))


def test_control_forwards_exact_saved_version_and_returns_confirmed_state(boundary):
    task, services, request = boundary
    updated = task.model_copy(update={"status": "paused", "version": 4})
    services.json_request.return_value = updated.model_dump(mode="json")
    result = asyncio.run(
        control_task(
            "alpha", task.id, "pause", TaskControl(expected_version=3), request
        )
    )
    assert result.status == "paused"
    services.json_request.assert_awaited_once_with(
        "Tsuchi",
        "POST",
        f"/api/agents/alpha/tasks/{task.id}/pause",
        json={"expected_version": 3},
    )


def test_control_cannot_confirm_an_unrelated_goal(boundary):
    task, services, request = boundary
    services.json_request.return_value = {
        **task.model_dump(mode="json"),
        "id": str(uuid4()),
    }
    with pytest.raises(ServiceError):
        asyncio.run(
            control_task(
                "alpha", task.id, "cancel", TaskControl(expected_version=3), request
            )
        )


@pytest.mark.parametrize("value", [0, "3", True])
def test_controls_require_a_positive_integer_saved_version(value):
    with pytest.raises(ValidationError):
        TaskControl(expected_version=value)
