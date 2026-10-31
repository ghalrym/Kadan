"""Codex completion events with mocked computer and storage boundaries."""

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4, uuid5

import pytest

from kadan.api.config import ServiceError
from kadan.api.pydantic_models.coding import CodexFinished, CodexOrigin
from kadan.runner.codex_events import CodexEvents


def finished(origin=True, status="completed"):
    return CodexFinished(
        request_id=uuid4(),
        thread_id=uuid4(),
        cwd="/home/gem/hello",
        origin=CodexOrigin(agent_key="guide", conversation_id=uuid4(), run_id=uuid4())
        if origin
        else None,
        status=status,
        message="Built the server.",
    )


@pytest.fixture
def boundary(monkeypatch):
    state = SimpleNamespace(finished=[], acknowledged=[], posts=[], error=None)

    async def read_finished(self):
        return state.finished

    async def acknowledge(self, request_id):
        state.acknowledged.append(request_id)

    async def json_request(service, method, path, json=None):
        state.posts.append((service, method, path, json))
        if state.error:
            raise state.error

    monkeypatch.setattr(
        "kadan.runner.codex_events.CodingClient.finished", read_finished
    )
    monkeypatch.setattr(
        "kadan.runner.codex_events.CodingClient.acknowledge", acknowledge
    )
    state.events = CodexEvents(
        SimpleNamespace(
            sandbox=SimpleNamespace(configured=True), json_request=json_request
        )
    )
    return state


def test_finished_run_queues_an_event_run_then_acknowledges(boundary):
    run = finished()
    boundary.finished = [run]
    asyncio.run(boundary.events.deliver())
    service, method, path, body = boundary.posts[0]
    assert (service, method) == ("Tsuchi", "POST")
    assert path == (
        f"/api/agents/guide/resources/{run.origin.conversation_id}/event-runs"
    )
    assert body["id"] == str(uuid5(run.request_id, "codex-event"))
    assert body["origin_run_id"] == str(run.origin.run_id)
    assert body["event"]["kind"] == "codex"
    assert json.loads(body["event"]["content"]) == {
        "thread_id": str(run.thread_id),
        "status": "completed",
        "cwd": "/home/gem/hello",
        "message": "Built the server.",
    }
    assert boundary.acknowledged == [run.request_id]


def test_storage_outage_leaves_the_run_for_the_next_poll(boundary):
    boundary.finished = [finished()]
    boundary.error = ServiceError("Tsuchi is unavailable.", 503)
    with pytest.raises(ServiceError):
        asyncio.run(boundary.events.deliver())
    assert boundary.acknowledged == []


@pytest.mark.parametrize(
    "status,message",
    [
        (404, "Conversation is gone."),
        (409, "Restore this item from trash before editing it"),
    ],
)
def test_missing_conversation_drops_the_result(boundary, status, message):
    run = finished()
    boundary.finished = [run]
    boundary.error = ServiceError(message, status)
    asyncio.run(boundary.events.deliver())
    assert boundary.acknowledged == [run.request_id]


def test_run_without_an_origin_is_acknowledged_without_an_event(boundary):
    run = finished(origin=False)
    boundary.finished = [run]
    asyncio.run(boundary.events.deliver())
    assert boundary.posts == []
    assert boundary.acknowledged == [run.request_id]


def test_conflicting_event_is_retained(boundary):
    boundary.finished = [finished()]
    boundary.error = ServiceError("This agent run ID is already in use", 409)
    with pytest.raises(ServiceError):
        asyncio.run(boundary.events.deliver())
    assert boundary.acknowledged == []


def test_one_failed_event_does_not_starve_other_conversations(boundary):
    failed, other = finished(), finished(origin=False)
    boundary.finished = [failed, other]
    boundary.error = ServiceError("Conflicting event", 409)
    with pytest.raises(ServiceError):
        asyncio.run(boundary.events.deliver())
    assert boundary.acknowledged == [other.request_id]
