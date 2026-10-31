import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from kadan.api.config import ServiceError
from kadan.api.pydantic_models.agent_runs import AgentRun, ImageJob, RunMessage
from kadan.api.pydantic_models.conversation_work import AgentRunSummary, ImageJobSummary
from kadan.api.pydantic_models.resources import Message, Resource
from kadan.api.routes.agents.conversations.runs import (
    RunInput,
    cancel_image,
    cancel_run,
    conversation_runs,
    create_run,
)
from kadan.api.services import LocalServices


@pytest.mark.parametrize(
    "body",
    [
        {"content": "Hi", "expected_version": 1},
        {"content": " ", "expected_version": 1, "persona_id": None},
        {
            "content": "Hi",
            "expected_version": 1,
            "persona_id": None,
            "tools": [{"type": "shell"}],
        },
    ],
)
def test_run_requests_reject_unsupported_or_ambiguous_input(body):
    with pytest.raises(ValidationError):
        RunInput.model_validate(body)


@pytest.fixture
def polling_boundary():
    now = datetime(2026, 10, 25, tzinfo=UTC)
    conversation = Resource(
        id=uuid4(),
        agent_key="alpha",
        kind="conversation",
        title="Chat",
        version=4,
        created_at=now,
        updated_at=now,
        messages=[Message(role="user", content="Draw a tree")],
    )
    run = AgentRun(
        id=uuid4(),
        agent_key="alpha",
        conversation_id=conversation.id,
        trigger_message_id=conversation.messages[0].id,
        trigger_job_id=uuid4(),
        delivered_message_ids=[uuid4()],
        status="running",
        owner_id=uuid4(),
        lease_expires_at=now + timedelta(minutes=1),
        version=3,
        created_at=now,
        updated_at=now + timedelta(seconds=5),
        transcript=[
            RunMessage(role="system", content="internal instructions"),
            RunMessage(role="user", content="Tool result: internal output"),
        ],
    )
    image = ImageJob(
        id=uuid4(),
        agent_key="alpha",
        run_id=run.id,
        conversation_id=conversation.id,
        request_id=uuid4(),
        event_run_id=uuid4(),
        prompt="A tree",
        status="running",
        upload_status="pending",
        owner_id=uuid4(),
        lease_expires_at=now + timedelta(minutes=1),
        version=2,
        created_at=now,
        updated_at=now + timedelta(seconds=10),
    )
    services = SimpleNamespace(
        resource=AsyncMock(return_value=conversation),
        runs=SimpleNamespace(
            enqueue=AsyncMock(return_value=run), cancel=AsyncMock(return_value=run)
        ),
    )
    services.resource_of_kinds = lambda *args: LocalServices.resource_of_kinds(
        services, *args
    )

    async def upstream(service, method, path, **kwargs):
        assert service == "Tsuchi"
        if path.endswith("/runs"):
            return [item.model_dump(mode="json") for item in services.run_records]
        if path.endswith("/image-jobs"):
            return [item.model_dump(mode="json") for item in services.image_records]
        if path.endswith("/cancel"):
            return image.model_dump(mode="json")
        raise AssertionError(path)

    services.run_records = [run]
    services.image_records = [image]
    services.json_request = AsyncMock(side_effect=upstream)
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(services=services))
    )
    return SimpleNamespace(
        conversation=conversation,
        run=run,
        image=image,
        services=services,
        request=request,
    )


def poll(boundary):
    return asyncio.run(
        conversation_runs("alpha", boundary.conversation.id, boundary.request)
    )


def test_poll_projects_only_display_fields_from_validated_records(polling_boundary):
    boundary = polling_boundary
    result = poll(boundary)
    assert isinstance(result.runs[0], AgentRunSummary)
    assert isinstance(result.images[0], ImageJobSummary)
    run_fields = {
        "id",
        "conversation_id",
        "trigger_message_id",
        "trigger_job_id",
        "delivered_message_ids",
        "status",
        "error",
        "created_at",
        "updated_at",
    }
    image_fields = {
        "id",
        "conversation_id",
        "prompt",
        "status",
        "error",
        "delivered_message_id",
        "created_at",
        "updated_at",
    }
    assert result.runs[0].model_dump() == boundary.run.model_dump(include=run_fields)
    assert result.images[0].model_dump() == boundary.image.model_dump(
        include=image_fields
    )
    assert set(result.model_dump()["runs"][0]) == run_fields
    assert set(result.model_dump()["images"][0]) == image_fields
    assert result.conversation == boundary.conversation
    serialized = result.model_dump_json()
    for excluded in (
        "transcript",
        "internal instructions",
        "internal output",
        "owner_id",
        "lease_expires_at",
        "upload_status",
        "request_id",
        "event_run_id",
        "asset_id",
        "trace_parent",
        "persona_id",
    ):
        assert excluded not in serialized


@pytest.mark.parametrize(
    "status", ["queued", "running", "completed", "failed", "cancelled"]
)
def test_poll_preserves_run_status_errors_and_delivery_receipts(
    polling_boundary, status
):
    boundary = polling_boundary
    changes = {
        "status": status,
        "error": "Recorded failure" if status == "failed" else None,
    }
    if status != "running":
        changes.update(owner_id=None, lease_expires_at=None)
    boundary.services.run_records = [
        AgentRun.model_validate(boundary.run.model_dump() | changes)
    ]
    result = poll(boundary).runs[0]
    assert result.status == status
    assert result.error == changes["error"]
    assert result.delivered_message_ids == boundary.run.delivered_message_ids


@pytest.mark.parametrize(
    "status", ["queued", "running", "completed", "failed", "cancelling", "cancelled"]
)
def test_poll_preserves_image_status_and_delivery_separately(polling_boundary, status):
    boundary = polling_boundary
    changes = {
        "status": status,
        "upload_status": "completed" if status == "completed" else "not_started",
        "asset_id": uuid4() if status == "completed" else None,
        "error": "Recorded image failure" if status == "failed" else None,
    }
    if status not in {"running", "cancelling"}:
        changes.update(owner_id=None, lease_expires_at=None)
    boundary.services.image_records = [
        ImageJob.model_validate(boundary.image.model_dump() | changes)
    ]
    result = poll(boundary).images[0]
    assert result.status == status
    assert result.error == changes["error"]
    assert result.delivered_message_id is None
    boundary.services.image_records[0].delivered_message_id = uuid4()
    assert (
        poll(boundary).images[0].delivered_message_id
        == boundary.services.image_records[0].delivered_message_id
    )


@pytest.mark.parametrize("record_type", ["run_records", "image_records"])
@pytest.mark.parametrize(
    "field,value", [("agent_key", "other"), ("conversation_id", uuid4())]
)
def test_poll_rejects_work_from_wrong_scope(
    polling_boundary, record_type, field, value
):
    setattr(getattr(polling_boundary.services, record_type)[0], field, value)
    with pytest.raises(ServiceError, match="outside this conversation") as caught:
        poll(polling_boundary)
    assert caught.value.status == 502
    assert caught.value.code == "invalid_response"


@pytest.mark.parametrize(
    "field,value,status",
    [
        ("agent_key", "other", 502),
        ("id", uuid4(), 502),
        ("kind", "space", 404),
        ("deleted_at", datetime.now(UTC), 404),
    ],
)
def test_poll_uses_active_conversation_read_boundary(
    polling_boundary, field, value, status
):
    conversation_id = polling_boundary.conversation.id
    setattr(polling_boundary.conversation, field, value)
    with pytest.raises(ServiceError) as caught:
        asyncio.run(
            conversation_runs("alpha", conversation_id, polling_boundary.request)
        )
    assert caught.value.status == status


@pytest.mark.parametrize("boundary_name", ["runs", "images", "conversation"])
def test_poll_propagates_upstream_failure_instead_of_empty_activity(
    polling_boundary, boundary_name
):
    failure = ServiceError("Tsuchi unavailable", 503)
    services = polling_boundary.services
    if boundary_name == "conversation":
        services.resource.side_effect = failure
    else:
        original = services.json_request.side_effect

        async def failing(service, method, path, **kwargs):
            if path.endswith("/runs" if boundary_name == "runs" else "/image-jobs"):
                raise failure
            return await original(service, method, path, **kwargs)

        services.json_request.side_effect = failing
    with pytest.raises(ServiceError) as caught:
        poll(polling_boundary)
    assert caught.value is failure


@pytest.mark.parametrize("record_type", ["run_records", "image_records"])
def test_poll_validates_full_record_before_summary(polling_boundary, record_type):
    getattr(polling_boundary.services, record_type)[0].owner_id = None
    with pytest.raises(ValidationError, match="owner and lease"):
        poll(polling_boundary)


def test_poll_keeps_run_creation_order_with_identity_tiebreak(polling_boundary):
    boundary = polling_boundary
    first = boundary.run.model_copy(update={"id": UUID(int=1)})
    second = boundary.run.model_copy(update={"id": UUID(int=2)})
    later = boundary.run.model_copy(
        update={
            "id": UUID(int=3),
            "created_at": boundary.run.created_at + timedelta(seconds=1),
        }
    )
    boundary.services.run_records = [later, second, first]
    assert [run.id for run in poll(boundary).runs] == [first.id, second.id, later.id]


def test_create_run_still_returns_full_record(polling_boundary):
    boundary = polling_boundary
    value = RunInput(content="Draw a tree", expected_version=4, persona_id=None)
    result = asyncio.run(
        create_run("alpha", boundary.conversation.id, value, boundary.request)
    )
    assert isinstance(result.run, AgentRun)
    assert result.run.model_dump() == boundary.run.model_dump()
    assert result.conversation == boundary.conversation
    boundary.services.runs.enqueue.assert_awaited_once_with(
        "alpha", boundary.conversation.id, value
    )


def test_cancel_run_still_returns_full_record(polling_boundary):
    boundary = polling_boundary
    result = asyncio.run(cancel_run("alpha", boundary.run.id, boundary.request))
    assert isinstance(result, AgentRun)
    assert result.model_dump() == boundary.run.model_dump()
    boundary.services.runs.cancel.assert_awaited_once_with("alpha", boundary.run.id)


def test_cancel_image_still_returns_full_record(polling_boundary):
    boundary = polling_boundary
    result = asyncio.run(cancel_image("alpha", boundary.image.id, boundary.request))
    assert isinstance(result, ImageJob)
    assert result.model_dump() == boundary.image.model_dump()
    boundary.services.json_request.assert_awaited_once_with(
        "Tsuchi",
        "POST",
        f"/api/agents/alpha/image-jobs/{boundary.image.id}/cancel",
        json={},
    )


@pytest.mark.parametrize("collection", ["runs", "image-jobs"])
@pytest.mark.parametrize("value", [{}, "", None, 0])
def test_poll_rejects_malformed_collections(polling_boundary, collection, value):
    services = polling_boundary.services
    original = services.json_request.side_effect

    async def malformed(service, method, path, **kwargs):
        if path.endswith("/" + collection):
            return value
        return await original(service, method, path, **kwargs)

    services.json_request.side_effect = malformed
    with pytest.raises(ServiceError, match="invalid work collection") as caught:
        poll(polling_boundary)
    assert caught.value.status == 502
    assert caught.value.code == "invalid_response"


def test_poll_accepts_valid_empty_collections(polling_boundary):
    polling_boundary.services.run_records = []
    polling_boundary.services.image_records = []
    result = poll(polling_boundary)
    assert result.runs == []
    assert result.images == []
    assert result.conversation == polling_boundary.conversation
