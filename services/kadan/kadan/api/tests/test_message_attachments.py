import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from kadan.api.config import ServiceError
from kadan.api.pydantic_models.resources import Resource
from kadan.api.routes.agents.conversations.notes import NoteInput
from kadan.api.routes.agents.conversations.runs import RunInput
from kadan.api.run_client import RunClient
from kadan.api.services import LocalServices


@pytest.mark.parametrize("model", [NoteInput, RunInput])
def test_attachment_list_preserves_order_and_defaults_to_no_files(model):
    extra = {"persona_id": None} if model is RunInput else {}
    ids = [uuid4(), uuid4()]
    assert (
        model(
            content="See files", expected_version=1, attachment_ids=ids, **extra
        ).attachment_ids
        == ids
    )
    assert model(content="A note", expected_version=1, **extra).attachment_ids == []


@pytest.mark.parametrize("model", [NoteInput, RunInput])
@pytest.mark.parametrize(
    "ids", [["not-an-id"], [uuid4()] * 2, [uuid4() for _ in range(101)]]
)
def test_invalid_duplicate_and_excess_attachments_are_rejected(model, ids):
    extra = {"persona_id": None} if model is RunInput else {}
    with pytest.raises(ValidationError):
        model(content="See files", expected_version=1, attachment_ids=ids, **extra)


@pytest.mark.parametrize("model", [NoteInput, RunInput])
def test_attachments_do_not_silently_enable_blank_messages(model):
    extra = {"persona_id": None} if model is RunInput else {}
    with pytest.raises(ValidationError):
        model(content=" ", expected_version=1, attachment_ids=[uuid4()], **extra)


def test_note_keeps_attachment_ids_without_contacting_inference():
    now = datetime.now(UTC)
    current = Resource(
        id=uuid4(),
        agent_key="ember",
        kind="conversation",
        title="Chat",
        version=1,
        created_at=now,
        updated_at=now,
    )
    services = object.__new__(LocalServices)
    services.resource = AsyncMock(return_value=current)
    services.append = AsyncMock(return_value=current)
    value = NoteInput(
        content="Keep these files",
        expected_version=1,
        attachment_ids=[uuid4(), uuid4()],
    )
    asyncio.run(services.save_note("ember", current.id, value))
    key, conversation_id, version, message = services.append.call_args.args
    assert (key, conversation_id, version) == ("ember", current.id, 1)
    assert message.id == value.message_id
    assert message.attachment_ids == value.attachment_ids


def test_run_forwards_attachments_in_atomic_storage_request_and_propagates_rejection():
    failure = ServiceError("Attachment missing or belongs to another agent", 422)
    storage = SimpleNamespace(json_request=AsyncMock(side_effect=failure))
    client = RunClient(storage)
    value = RunInput(
        content="Use these files",
        expected_version=2,
        persona_id=None,
        attachment_ids=[uuid4(), uuid4()],
    )
    conversation_id = uuid4()
    with pytest.raises(ServiceError, match="Attachment missing"):
        asyncio.run(client.enqueue("ember", conversation_id, value))
    args = storage.json_request.call_args
    assert args.args == (
        "Tsuchi",
        "POST",
        f"/api/agents/ember/resources/{conversation_id}/runs",
    )
    assert args.kwargs["json"]["message"]["attachment_ids"] == [
        str(id) for id in value.attachment_ids
    ]
    storage.json_request.assert_awaited_once()


def test_note_propagates_transactional_attachment_rejection_without_retry():
    now = datetime.now(UTC)
    current = Resource(
        id=uuid4(),
        agent_key="ember",
        kind="conversation",
        title="Chat",
        version=1,
        created_at=now,
        updated_at=now,
    )
    services = object.__new__(LocalServices)
    services.resource = AsyncMock(return_value=current)
    services.append = AsyncMock(side_effect=ServiceError("Attachment is in trash", 409))
    value = NoteInput(content="Keep this", expected_version=1, attachment_ids=[uuid4()])
    with pytest.raises(ServiceError, match="in trash"):
        asyncio.run(services.save_note("ember", current.id, value))
    services.append.assert_awaited_once()
