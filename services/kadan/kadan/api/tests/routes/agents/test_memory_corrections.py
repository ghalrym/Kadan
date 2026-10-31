import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from kadan.api.config import ServiceError, Settings
from kadan.api.pydantic_models.resources import Memory, Resource
from kadan.api.routes.agents.memories import MemoryUpdate, update_memory
from kadan.api.services import LocalServices

NOW = datetime(2026, 10, 25, tzinfo=UTC)


def memory(**changes):
    return Resource.model_validate(
        {
            "id": uuid4(),
            "agent_key": "alpha",
            "kind": "memory",
            "title": "Old title",
            "markdown": "Old memory",
            "version": 3,
            "created_at": NOW,
            "updated_at": NOW,
            "parent_id": uuid4(),
            "tags": ["preference"],
            "attachment_ids": [uuid4()],
            "sources": [
                {"label": "Original conversation", "url": "https://example.org/source"}
            ],
        }
        | changes
    )


def update(**changes):
    return MemoryUpdate.model_validate(
        {
            "title": "Corrected title",
            "markdown": "Corrected memory",
            "expected_version": 3,
        }
        | changes
    )


def boundary(current, result=None):
    service = LocalServices(AsyncMock(), Settings(_env_file=None))
    service.resource = AsyncMock(return_value=current)
    result = result or current.model_copy(
        update={
            "title": "Corrected title",
            "markdown": "Corrected memory",
            "version": 4,
        }
    )
    service.json_request = AsyncMock(return_value=result.model_dump(mode="json"))
    return service


@pytest.mark.parametrize(
    "changes",
    [
        {"title": "  "},
        {"title": "x" * 501},
        {"markdown": "x" * 2000001},
        {"expected_version": 0},
        {"tags": ["cannot change metadata"]},
        {"kind": "space"},
    ],
)
def test_correction_input_limits_writes_to_title_markdown_and_version(changes):
    with pytest.raises(ValidationError):
        update(**changes)


def test_correction_preserves_existing_metadata_and_sends_expected_version():
    current = memory()
    service = boundary(current)
    result = asyncio.run(
        service.update_memory("alpha", current.id, update(title="  Corrected title  "))
    )
    assert isinstance(result, Memory)
    body = service.json_request.await_args.kwargs["json"]
    assert body == current.model_dump(
        mode="json",
        exclude={
            "id",
            "agent_key",
            "version",
            "created_at",
            "updated_at",
            "deleted_at",
        },
    ) | {
        "title": "Corrected title",
        "markdown": "Corrected memory",
        "expected_version": 3,
    }
    assert result.parent_id == current.parent_id
    assert result.sources == current.sources
    assert result.attachment_ids == current.attachment_ids
    service.json_request.assert_awaited_once()


@pytest.mark.parametrize(
    "changes,status",
    [
        ({"kind": "space"}, 404),
        ({"deleted_at": NOW}, 404),
        ({"agent_key": "other"}, 502),
        ({"version": 4}, 409),
    ],
)
def test_ineligible_or_newer_memory_is_not_written(changes, status):
    current = memory(**changes)
    service = boundary(current)
    with pytest.raises(ServiceError) as error:
        asyncio.run(service.update_memory("alpha", current.id, update()))
    assert error.value.status == status
    service.json_request.assert_not_awaited()


def test_wrong_upstream_id_is_not_written():
    service = boundary(memory())
    with pytest.raises(ServiceError) as error:
        asyncio.run(service.update_memory("alpha", uuid4(), update()))
    assert error.value.status == 502
    service.json_request.assert_not_awaited()


@pytest.mark.parametrize(
    "changes",
    [
        {"id": uuid4()},
        {"agent_key": "other"},
        {"kind": "space"},
        {"deleted_at": NOW},
        {"version": 3},
        {"title": "Different"},
        {"markdown": "Different"},
    ],
)
def test_unconfirmed_receipt_does_not_report_success(changes):
    current = memory()
    receipt = Resource.model_validate(
        current.model_dump()
        | {"title": "Corrected title", "markdown": "Corrected memory", "version": 4}
        | changes
    )
    service = boundary(current, receipt)
    with pytest.raises(ServiceError, match="Check the saved memory") as error:
        asyncio.run(service.update_memory("alpha", current.id, update()))
    assert error.value.status == 502
    service.json_request.assert_awaited_once()


@pytest.mark.parametrize("status", [409, 503])
def test_conflict_or_uncertain_write_is_returned_without_retry(status):
    current = memory()
    service = boundary(current)
    failure = ServiceError("Storage response was lost", status)
    service.json_request.side_effect = failure
    with pytest.raises(ServiceError) as error:
        asyncio.run(service.update_memory("alpha", current.id, update()))
    assert error.value is failure
    service.json_request.assert_awaited_once()


def test_route_uses_memory_service_without_inference_or_runner_work():
    current = memory()
    service = SimpleNamespace(update_memory=AsyncMock(return_value=current))
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(services=service))
    )
    value = update()
    assert asyncio.run(update_memory("alpha", current.id, value, request)) == current
    service.update_memory.assert_awaited_once_with("alpha", current.id, value)


def test_title_only_correction_preserves_a_large_existing_memory():
    current = memory(markdown="x" * 2000000)
    receipt = current.model_copy(update={"title": "Corrected title", "version": 4})
    service = boundary(current, receipt)
    result = asyncio.run(
        service.update_memory("alpha", current.id, update(markdown=current.markdown))
    )
    assert result.markdown == current.markdown
    assert (
        service.json_request.await_args.kwargs["json"]["markdown"] == current.markdown
    )
