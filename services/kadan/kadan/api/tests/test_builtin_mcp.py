import asyncio
import json
from copy import deepcopy
from uuid import UUID

import pytest
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import ValidationError

from kadan.api.builtin_mcp import create_builtin_server
from kadan.api.config import ServiceError

RESOURCE_ID = UUID("10000000-0000-4000-8000-000000000001")
PARENT_ID = UUID("10000000-0000-4000-8000-000000000002")
NOW = "2026-10-17T12:00:00Z"


def document(**changes):
    return {
        "id": str(RESOURCE_ID),
        "agent_key": "andrew",
        "kind": "memory",
        "title": "A preference",
        "markdown": "Use concise prose.",
        "version": 2,
        "created_at": NOW,
        "updated_at": NOW,
        "deleted_at": None,
        **changes,
    }


class Storage:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    async def json_request(self, service, method, path, **kwargs):
        self.requests.append((service, method, path, deepcopy(kwargs)))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return deepcopy(response)


def call(storage, name, **arguments):
    server = create_builtin_server(storage, "andrew")
    blocks = asyncio.run(server.call_tool(name, arguments))
    return json.loads(blocks[0].text)


def test_tools_have_bound_namespace_and_explicit_schemas():
    tools = asyncio.run(create_builtin_server(Storage(), "andrew").list_tools())
    assert {tool.name for tool in tools} == {
        f"{kind}_{operation}"
        for kind in ("memory", "workspace")
        for operation in ("search", "read", "create", "update")
    }
    for tool in tools:
        assert tool.inputSchema["additionalProperties"] is False
        assert "agent_key" not in tool.inputSchema["properties"]
        assert tool.annotations.readOnlyHint == tool.name.endswith(("search", "read"))


@pytest.mark.parametrize("key", ["../other", "UPPER", "a/b", "", 4])
def test_invalid_namespace_never_reaches_storage(key):
    with pytest.raises(ValidationError):
        create_builtin_server(Storage(), key)


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("memory_search", {"agent_key": "other"}),
        ("workspace_read", {"resource_id": "../../other"}),
        ("memory_search", {"limit": True}),
        ("memory_search", {"limit": "2"}),
        ("memory_search", {"offset": -1}),
        ("memory_search", {"limit": 21}),
        ("memory_search", {"query": "x" * 501}),
        ("workspace_read", {"resource_id": str(RESOURCE_ID), "limit": 12001}),
        ("memory_create", {"title": "   ", "markdown": ""}),
        ("memory_create", {"title": "Title", "markdown": "x" * 48001}),
        ("memory_create", {"title": "Title", "markdown": "", "tags": '["a"]'}),
        (
            "memory_update",
            {"resource_id": str(RESOURCE_ID), "title": "Title", "markdown": ""},
        ),
        (
            "memory_update",
            {
                "resource_id": str(RESOURCE_ID),
                "title": "Title",
                "markdown": "",
                "expected_version": False,
            },
        ),
    ],
)
def test_invalid_arguments_fail_before_storage(name, arguments):
    storage = Storage()
    with pytest.raises(ToolError):
        call(storage, name, **arguments)
    assert storage.requests == []


@pytest.mark.parametrize(
    ("prefix", "kind"), [("memory", "memory"), ("workspace", "space")]
)
def test_search_uses_tsuchi_namespace_kind_query_and_pagination(prefix, kind):
    storage = Storage([document(kind=kind, markdown="a" * 2000), document(kind=kind)])
    result = call(storage, f"{prefix}_search", query="preference", offset=5, limit=1)
    assert storage.requests == [
        (
            "Tsuchi",
            "GET",
            "/api/agents/andrew/resources",
            {
                "params": {
                    "kind": kind,
                    "q": "preference",
                    "trash": "false",
                    "offset": 5,
                    "limit": 2,
                }
            },
        )
    ]
    assert result["next_offset"] == 6
    assert len(result["items"]) == 1
    assert result["items"][0]["excerpt"] == "a" * 300
    assert result["items"][0]["total_characters"] == 2000
    assert result["items"][0]["version"] == 2


def test_search_skips_trash_without_losing_page_position():
    storage = Storage([document(deleted_at=NOW), document()])
    result = call(storage, "memory_search", limit=1)
    assert result == {"items": [], "next_offset": 1}


@pytest.mark.parametrize("changes", [{"agent_key": "other"}, {"kind": "space"}])
def test_search_rejects_out_of_scope_results(changes):
    with pytest.raises(ToolError):
        call(Storage([document(**changes)]), "memory_search")


@pytest.mark.parametrize("value", [{}, [document()] * 12, [None]])
def test_search_rejects_invalid_upstream_page(value):
    with pytest.raises(ToolError, match="invalid"):
        call(Storage(value), "memory_search")


def test_read_pages_markdown_with_snapshot_version():
    storage = Storage(document(markdown="0123456789"))
    result = call(
        storage, "memory_read", resource_id=str(RESOURCE_ID), offset=2, limit=3
    )
    assert result["markdown"] == "234"
    assert result["offset"] == 2
    assert result["next_offset"] == 5
    assert result["total_characters"] == 10
    assert result["version"] == 2
    assert storage.requests[0][2].endswith(str(RESOURCE_ID))


def test_read_last_page_has_no_next_offset():
    result = call(
        Storage(document()), "memory_read", resource_id=str(RESOURCE_ID), offset=100
    )
    assert result["markdown"] == ""
    assert result["next_offset"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"agent_key": "other"},
        {"kind": "space"},
        {"deleted_at": NOW},
        {"id": str(PARENT_ID)},
    ],
)
@pytest.mark.parametrize("operation", ["read", "update"])
def test_read_and_update_reject_inactive_or_out_of_scope_resource(changes, operation):
    storage = Storage(document(**changes))
    arguments = {"resource_id": str(RESOURCE_ID)}
    if operation == "update":
        arguments.update(expected_version=2, title="Edit", markdown="Revised")
    with pytest.raises(ToolError):
        call(storage, f"memory_{operation}", **arguments)
    assert len(storage.requests) == 1


@pytest.mark.parametrize(
    ("prefix", "kind"), [("memory", "memory"), ("workspace", "space")]
)
def test_create_stores_document_in_bound_namespace(prefix, kind):
    storage = Storage(document(kind=kind, version=1))
    result = call(
        storage,
        f"{prefix}_create",
        title=" New document ",
        markdown="Content",
        tags=["project"],
    )
    assert storage.requests == [
        (
            "Tsuchi",
            "POST",
            "/api/agents/andrew/resources",
            {
                "json": {
                    "kind": kind,
                    "title": "New document",
                    "markdown": "Content",
                    "tags": ["project"],
                    "parent_id": None,
                }
            },
        )
    ]
    assert result["id"] == str(RESOURCE_ID)
    assert result["version"] == 1


def test_create_checks_parent_scope_and_kind():
    storage = Storage(document(id=str(PARENT_ID), agent_key="other", kind="space"))
    with pytest.raises(ToolError, match="namespace"):
        call(
            storage,
            "workspace_create",
            title="Child",
            markdown="",
            parent_id=str(PARENT_ID),
        )
    assert len(storage.requests) == 1


def test_create_accepts_active_space_parent():
    storage = Storage(document(id=str(PARENT_ID), kind="space"), document(kind="space"))
    call(
        storage,
        "workspace_create",
        title="Child",
        markdown="",
        parent_id=str(PARENT_ID),
    )
    assert storage.requests[1][3]["json"]["parent_id"] == str(PARENT_ID)


def test_update_preserves_metadata_and_sends_expected_version():
    original = document(
        kind="space",
        parent_id=str(PARENT_ID),
        tags=["project"],
        sources=[{"label": "Reference", "url": "https://example.com"}],
        attachment_ids=[str(PARENT_ID)],
    )
    storage = Storage(original, document(kind="space", version=3))
    result = call(
        storage,
        "workspace_update",
        resource_id=str(RESOURCE_ID),
        expected_version=2,
        title="Edited",
        markdown="Replacement",
    )
    payload = storage.requests[1][3]["json"]
    assert payload == {
        "kind": "space",
        "title": "Edited",
        "markdown": "Replacement",
        "parent_id": str(PARENT_ID),
        "tags": ["project"],
        "sources": original["sources"],
        "attachment_ids": [str(PARENT_ID)],
        "deck": None,
        "messages": [],
        "expected_version": 2,
    }
    assert result["version"] == 3


def test_stale_version_does_not_write():
    storage = Storage(document(version=3))
    with pytest.raises(ToolError, match="changed"):
        call(
            storage,
            "memory_update",
            resource_id=str(RESOURCE_ID),
            expected_version=2,
            title="Edited",
            markdown="Replacement",
        )
    assert len(storage.requests) == 1


def test_tsuchi_cas_race_returns_readable_error():
    storage = Storage(
        document(),
        ServiceError(
            "This item changed. Reload it before saving again.", 409, "version_conflict"
        ),
    )
    with pytest.raises(ToolError, match="This item changed"):
        call(
            storage,
            "memory_update",
            resource_id=str(RESOURCE_ID),
            expected_version=2,
            title="Edited",
            markdown="Replacement",
        )
    assert storage.requests[1][3]["json"]["expected_version"] == 2


def test_storage_failure_is_readable():
    with pytest.raises(ToolError, match="Tsuchi is unavailable"):
        call(Storage(ServiceError("Tsuchi is unavailable")), "memory_search")


def test_cancellation_propagates_without_retry_or_write():
    storage = Storage(asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        call(
            storage,
            "memory_update",
            resource_id=str(RESOURCE_ID),
            expected_version=2,
            title="Edit",
            markdown="Edit",
        )
    assert len(storage.requests) == 1


@pytest.mark.parametrize("operation", ["create", "update"])
@pytest.mark.parametrize(
    "response",
    [
        ServiceError("Tsuchi is unavailable."),
        ServiceError("Tsuchi returned HTTP 500.", 500, "upstream_error"),
        ServiceError("Tsuchi returned an invalid response.", 502, "invalid_response"),
        {},
        document(agent_key="other"),
        document(kind="space"),
        document(deleted_at=NOW),
    ],
)
def test_uncertain_write_warns_before_retry(operation, response):
    storage = Storage(*([document()] if operation == "update" else []), response)
    arguments = {"title": "Edited", "markdown": "Replacement"}
    if operation == "update":
        arguments.update(resource_id=str(RESOURCE_ID), expected_version=2)
    with pytest.raises(ToolError, match="write may already have completed") as failure:
        call(storage, f"memory_{operation}", **arguments)
    assert "Read or search the current state before retrying" in str(failure.value)
    assert len(storage.requests) == (2 if operation == "update" else 1)


@pytest.mark.parametrize("operation", ["create", "update"])
@pytest.mark.parametrize("status", [409, 422])
def test_definite_write_rejection_does_not_claim_uncertain_outcome(operation, status):
    rejection = ServiceError("Tsuchi rejected this change.", status, "upstream_error")
    storage = Storage(*([document()] if operation == "update" else []), rejection)
    arguments = {"title": "Edited", "markdown": "Replacement"}
    if operation == "update":
        arguments.update(resource_id=str(RESOURCE_ID), expected_version=2)
    with pytest.raises(ToolError, match="Tsuchi rejected this change") as failure:
        call(storage, f"memory_{operation}", **arguments)
    assert "may already have completed" not in str(failure.value)


def test_failed_read_before_update_does_not_claim_write_started():
    storage = Storage(ServiceError("Tsuchi is unavailable."))
    with pytest.raises(ToolError) as failure:
        call(
            storage,
            "memory_update",
            resource_id=str(RESOURCE_ID),
            expected_version=2,
            title="Edited",
            markdown="Replacement",
        )
    assert "may already have completed" not in str(failure.value)
    assert len(storage.requests) == 1
    assert storage.requests[0][1] == "GET"
