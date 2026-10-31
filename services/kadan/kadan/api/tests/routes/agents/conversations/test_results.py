import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4, uuid5

import pytest
from pydantic import ValidationError

from kadan.api.config import ServiceError
from kadan.api.pydantic_models.resources import Resource
from kadan.api.routes.agents.conversations.results import SaveResultInput, save_result


@pytest.mark.parametrize(
    "body",
    [{"title": " "}, {"title": "x" * 501}, {"title": "OK", "markdown": "Override"}],
)
def test_result_title_rejects_blank_overlong_and_untrusted_content(body):
    with pytest.raises(ValidationError):
        SaveResultInput.model_validate(body)


def test_result_proxy_forwards_only_title_and_saved_source_ids():
    conversation_id, message_id = uuid4(), uuid4()
    now = datetime.now(UTC)
    document = Resource(
        id=uuid5(conversation_id, f"chat-result:{message_id}"),
        agent_key="alpha",
        kind="space",
        title="Saved title",
        version=2,
        created_at=now,
        updated_at=now,
    )
    services = SimpleNamespace(
        json_request=AsyncMock(return_value=document.model_dump())
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(services=services))
    )
    result = asyncio.run(
        save_result(
            "alpha",
            conversation_id,
            message_id,
            SaveResultInput(title=" Title "),
            request,
        )
    )
    assert result.id == document.id
    services.json_request.assert_awaited_once_with(
        "Tsuchi",
        "POST",
        f"/api/agents/alpha/resources/{conversation_id}/messages/{message_id}/document",
        json={"title": "Title"},
    )


@pytest.mark.parametrize(
    "change",
    [
        {"agent_key": "beta"},
        {"id": uuid4()},
        {"deleted_at": datetime.now(UTC)},
        {"kind": "memory"},
        {"kind": "deck"},
    ],
)
def test_result_proxy_does_not_confirm_an_unexpected_storage_record(change):
    conversation_id, message_id = uuid4(), uuid4()
    now = datetime.now(UTC)
    record = (
        Resource(
            id=uuid5(conversation_id, f"chat-result:{message_id}"),
            agent_key="alpha",
            kind="space",
            title="Result",
            version=1,
            created_at=now,
            updated_at=now,
        ).model_dump()
        | change
    )
    services = SimpleNamespace(json_request=AsyncMock(return_value=record))
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(services=services))
    )
    with pytest.raises(ServiceError, match="invalid saved result"):
        asyncio.run(
            save_result(
                "alpha",
                conversation_id,
                message_id,
                SaveResultInput(title="Result"),
                request,
            )
        )
