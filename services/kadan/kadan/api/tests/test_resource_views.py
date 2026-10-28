import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from kadan.api.config import ServiceError, Settings
from kadan.api.pydantic_models.resources import (
    Memory,
    Resource,
    WorkspaceDocument,
)
from kadan.api.routes.agents import memories, workspace
from kadan.api.services import LocalServices


def record(kind="space", **changes):
    now = datetime(2026, 10, 22, tzinfo=UTC)
    return Resource(
        **(
            {
                "id": uuid4(),
                "agent_key": "alpha",
                "kind": kind,
                "title": "Saved record",
                "version": 3,
                "created_at": now,
                "updated_at": now,
            }
            | changes
        )
    )


def run(value, operation):
    async def check():
        def respond(request):
            return httpx.Response(200, json=value)

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await operation(LocalServices(client, Settings(_env_file=None)))

    return asyncio.run(check())


@pytest.mark.parametrize("kind", ["memory", "persona", "conversation"])
def test_memory_and_internal_records_are_not_document_models(kind):
    with pytest.raises(ValidationError):
        WorkspaceDocument.model_validate(record(kind).model_dump())


@pytest.mark.parametrize("kind", ["space", "deck", "persona", "conversation"])
def test_non_memory_records_are_not_memory_models(kind):
    with pytest.raises(ValidationError):
        Memory.model_validate(record(kind).model_dump())


@pytest.mark.parametrize(
    ("routes", "model", "kind", "kinds"),
    [
        (workspace, WorkspaceDocument, "space", workspace.DOCUMENT_KINDS),
        (memories, Memory, "memory", memories.MEMORY_KINDS),
    ],
)
def test_read_routes_preserve_record_metadata_and_collection_selection(
    routes, model, kind, kinds
):
    saved = record(kind, parent_id=uuid4(), tags=["saved"], markdown="Original content")
    service = SimpleNamespace(
        list_resource_kinds=AsyncMock(return_value=[saved]),
        resource_of_kinds=AsyncMock(return_value=saved),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(services=service))
    )
    listing = (
        workspace.list_documents if routes is workspace else memories.list_memories
    )
    detail = workspace.get_document if routes is workspace else memories.get_memory
    listed = asyncio.run(listing("alpha", request, q="text", limit=10, offset=20))
    opened = asyncio.run(detail("alpha", saved.id, request))
    assert isinstance(listed[0], model)
    assert opened.model_dump() == saved.model_dump()
    service.list_resource_kinds.assert_awaited_once_with(
        "alpha", kinds, q="text", limit=10, offset=20
    )
    service.resource_of_kinds.assert_awaited_once_with("alpha", saved.id, kinds)


def test_document_list_requests_combined_kinds_before_pagination():
    saved = record("deck")
    requests = []

    async def check():
        def respond(request):
            requests.append(request)
            return httpx.Response(200, json=[saved.model_dump(mode="json")])

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await LocalServices(
                client, Settings(_env_file=None)
            ).list_resource_kinds(
                "alpha", workspace.DOCUMENT_KINDS, q="notes", limit=7, offset=14
            )

    assert asyncio.run(check()) == [saved]
    assert requests[0].url.path == "/api/agents/alpha/resources"
    assert requests[0].url.params.get_list("kinds") == ["space", "deck"]
    assert dict(requests[0].url.params) | {"kinds": None} == {
        "kinds": None,
        "q": "notes",
        "limit": "7",
        "offset": "14",
        "trash": "false",
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "memory"},
        {"agent_key": "beta"},
        {"deleted_at": datetime(2026, 10, 22, tzinfo=UTC)},
    ],
)
def test_invalid_upstream_collection_fails_instead_of_hiding_rows(changes):
    value = record(**changes).model_dump(mode="json")
    with pytest.raises(ServiceError) as error:
        run(
            [value],
            lambda service: service.list_resource_kinds(
                "alpha", workspace.DOCUMENT_KINDS
            ),
        )
    assert error.value.status == 502
    assert error.value.code == "invalid_response"


@pytest.mark.parametrize("value", [{}, None, [{"title": "incomplete"}]])
def test_malformed_upstream_collection_is_an_explicit_error(value):
    with pytest.raises(ServiceError) as error:
        run(
            value,
            lambda service: service.list_resource_kinds(
                "alpha", workspace.DOCUMENT_KINDS
            ),
        )
    assert error.value.status == 502


@pytest.mark.parametrize(
    ("kind", "kinds", "deleted"),
    [
        ("memory", workspace.DOCUMENT_KINDS, False),
        ("space", memories.MEMORY_KINDS, False),
        ("space", workspace.DOCUMENT_KINDS, True),
        ("memory", memories.MEMORY_KINDS, True),
    ],
)
def test_detail_hides_other_collection_and_trash(kind, kinds, deleted):
    saved = record(
        kind, deleted_at=datetime(2026, 10, 22, tzinfo=UTC) if deleted else None
    )
    with pytest.raises(ServiceError) as error:
        run(
            saved.model_dump(mode="json"),
            lambda service: service.resource_of_kinds("alpha", saved.id, kinds),
        )
    assert error.value.status == 404


@pytest.mark.parametrize("changes", [{"agent_key": "beta"}, {"id": uuid4()}])
def test_detail_rejects_an_upstream_namespace_or_id_mismatch(changes):
    saved = record()
    value = saved.model_dump(mode="json") | changes
    value = Resource.model_validate(value).model_dump(mode="json")
    with pytest.raises(ServiceError) as error:
        run(
            value,
            lambda service: service.resource_of_kinds(
                "alpha", saved.id, workspace.DOCUMENT_KINDS
            ),
        )
    assert error.value.status == 502


def test_existing_memory_parent_link_is_preserved():
    saved = record("memory", parent_id=uuid4())
    assert (
        run(
            saved.model_dump(mode="json"),
            lambda service: service.resource_of_kinds(
                "alpha", saved.id, memories.MEMORY_KINDS
            ),
        )
        == saved
    )
