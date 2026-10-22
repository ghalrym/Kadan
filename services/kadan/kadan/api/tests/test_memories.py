"""Memory tools against recording storage boundaries; no inference or live storage."""

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from kadan.api.config import ServiceError, Settings
from kadan.api.services import LocalServices
from kadan.runner.agent_inference import ToolCommand
from kadan.runner.agent_runs import AgentRuns
from kadan.runner.memories import AgentMemories
from kadan.runner.tools import MEMORY_TOOLS
from kadan.runner.tools.memory import (
    CreateMemoryArguments,
    ListMemoriesArguments,
    ReadMemoryArguments,
    UpdateMemoryArguments,
)

MEMORY_ID = uuid4()
PARENT_ID = uuid4()
ASSET_ID = uuid4()
NOW = datetime(2026, 10, 22, tzinfo=UTC).isoformat()


def memory(**changes):
    return {
        "id": str(MEMORY_ID),
        "agent_key": "alpha",
        "kind": "memory",
        "title": "Preferences",
        "markdown": "Use concise answers.",
        "tags": ["preferences"],
        "version": 3,
        "created_at": NOW,
        "updated_at": NOW,
        **changes,
    }


def call(handler, action):
    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = LocalServices(client, Settings(_env_file=None))
            return await action(AgentMemories(service, "alpha"))

    return asyncio.run(execute())


def test_list_searches_only_active_agent_memories_before_pagination():
    def storage(request):
        assert request.url.path == "/api/agents/alpha/resources"
        assert dict(request.url.params) == {
            "kind": "memory",
            "q": "concise",
            "trash": "false",
            "limit": "2",
            "offset": "4",
        }
        return httpx.Response(200, json=[memory()])

    result = call(storage, lambda memories: memories.list("concise", 2, 4))
    assert result == {
        "memories": [
            {
                "id": str(MEMORY_ID),
                "title": "Preferences",
                "tags": ["preferences"],
                "version": 3,
            }
        ],
        "limit": 2,
        "offset": 4,
    }


def test_read_returns_content_and_version_for_updates():
    def storage(request):
        assert request.url.path == f"/api/agents/alpha/resources/{MEMORY_ID}"
        return httpx.Response(200, json=memory())

    result = call(storage, lambda memories: memories.read(MEMORY_ID))
    assert result["markdown"] == "Use concise answers."
    assert result["version"] == 3


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "space"},
        {"kind": "deck"},
        {"kind": "persona"},
        {"kind": "conversation"},
        {"agent_key": "beta"},
        {"deleted_at": NOW},
    ],
)
@pytest.mark.parametrize("operation", ["read", "update"])
def test_read_and_update_reject_documents_other_agents_and_trash(changes, operation):
    calls = []

    def storage(request):
        calls.append(request.method)
        return httpx.Response(200, json=memory(**changes))

    with pytest.raises(ServiceError):
        call(
            storage,
            lambda memories: (
                memories.read(MEMORY_ID)
                if operation == "read"
                else memories.update(MEMORY_ID, 3, "Changed", "New text", None)
            ),
        )
    assert calls == ["GET"]


def test_create_persists_a_memory_without_workspace_parent_or_other_resource_kind():
    def storage(request):
        assert request.method == "POST"
        assert request.url.path == "/api/agents/alpha/resources"
        body = json.loads(request.content)
        assert body == {
            "kind": "memory",
            "title": "Preferences",
            "markdown": "Short answers",
            "tags": [],
        }
        return httpx.Response(201, json=memory(**body, version=1))

    result = call(
        storage, lambda memories: memories.create("Preferences", "Short answers", [])
    )
    assert result["version"] == 1


@pytest.mark.parametrize("tags, expected_tags", [(None, ["preferences"]), ([], [])])
def test_update_preserves_other_metadata_and_passes_expected_version(
    tags, expected_tags
):
    calls = []
    existing = memory(
        parent_id=str(PARENT_ID),
        attachment_ids=[str(ASSET_ID)],
        sources=[{"label": "Chat", "url": "https://example.com/chat"}],
    )

    def storage(request):
        calls.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, json=existing)
        assert request.url.path == f"/api/agents/alpha/resources/{MEMORY_ID}"
        body = json.loads(request.content)
        assert body == {
            "kind": "memory",
            "title": "Updated",
            "markdown": "New text",
            "tags": expected_tags,
            "expected_version": 3,
            "parent_id": str(PARENT_ID),
            "attachment_ids": [str(ASSET_ID)],
            "sources": existing["sources"],
            "deck": None,
            "messages": [],
        }
        return httpx.Response(200, json={**existing, **body, "version": 4})

    result = call(
        storage,
        lambda memories: memories.update(MEMORY_ID, 3, "Updated", "New text", tags),
    )
    assert calls == ["GET", "PUT"]
    assert result["version"] == 4


def test_stale_update_does_not_write_or_refresh_the_expected_version():
    calls = []

    def storage(request):
        calls.append(request.method)
        return httpx.Response(200, json=memory())

    with pytest.raises(ServiceError) as error:
        call(
            storage,
            lambda memories: memories.update(MEMORY_ID, 2, "Updated", "Text", None),
        )
    assert error.value.code == "version_conflict"
    assert calls == ["GET"]


def test_concurrent_storage_conflict_is_not_retried():
    calls = []

    def storage(request):
        calls.append(request.method)
        return (
            httpx.Response(200, json=memory())
            if request.method == "GET"
            else httpx.Response(409, json={"detail": "Memory changed"})
        )

    with pytest.raises(ServiceError) as error:
        call(
            storage,
            lambda memories: memories.update(MEMORY_ID, 3, "Updated", "Text", None),
        )
    assert error.value.code == "version_conflict"
    assert calls == ["GET", "PUT"]


def test_uncertain_create_is_not_retried():
    calls = []

    def storage(request):
        calls.append(request.method)
        raise httpx.ReadTimeout("No response", request=request)

    with pytest.raises(ServiceError):
        call(storage, lambda memories: memories.create("Title", "Text", []))
    assert calls == ["POST"]


@pytest.mark.parametrize(
    "schema, arguments",
    [
        (ListMemoriesArguments, {"agent_key": "beta"}),
        (ListMemoriesArguments, {"limit": 0}),
        (ReadMemoryArguments, {"memory_id": str(MEMORY_ID), "agent_key": "beta"}),
        (
            CreateMemoryArguments,
            {"title": "Title", "markdown": "Text", "kind": "space"},
        ),
        (CreateMemoryArguments, {"title": " ", "markdown": "Text"}),
        (
            CreateMemoryArguments,
            {"title": "Title", "markdown": "Text", "parent_id": str(PARENT_ID)},
        ),
        (
            UpdateMemoryArguments,
            {"memory_id": str(MEMORY_ID), "title": "Title", "markdown": "Text"},
        ),
        (
            UpdateMemoryArguments,
            {
                "memory_id": str(MEMORY_ID),
                "title": "Title",
                "markdown": "Text",
                "expected_version": True,
            },
        ),
    ],
)
def test_tool_schemas_reject_scope_overrides_and_invalid_writes(schema, arguments):
    with pytest.raises(ValidationError):
        schema.model_validate(arguments)


@pytest.mark.parametrize("tool", MEMORY_TOOLS, ids=lambda tool: tool.name)
def test_runner_dispatches_memory_tools_without_computer_and_fences_replayed_writes(
    tool,
):
    controller = object.__new__(AgentRuns)
    controller.services = SimpleNamespace(json_request=AsyncMock(return_value=memory()))
    controller.save_result = AsyncMock()
    run = SimpleNamespace(id=uuid4(), transcript=[], agent_key="alpha")
    state = SimpleNamespace(
        run=run, command=ToolCommand(tool=tool.name, arguments={}), replayed=True
    )
    if tool.name == "list_memories":
        controller.services.json_request.return_value = [memory()]
    if tool.name == "read_memory":
        state.command.arguments = {"memory_id": str(MEMORY_ID)}
    asyncio.run(controller.invoke_tool(state))
    result = controller.save_result.call_args.args[2]
    if tool.metadata["method"] == "GET":
        assert "error" not in result
        assert controller.services.json_request.call_args.args[2].startswith(
            "/api/agents/alpha/resources"
        )
    else:
        assert "may or may not have saved" in result["error"]
        controller.services.json_request.assert_not_called()


def test_runner_returns_storage_errors_as_tool_results():
    controller = object.__new__(AgentRuns)
    controller.services = SimpleNamespace(
        json_request=AsyncMock(side_effect=ServiceError("Storage unavailable"))
    )
    tool = next(tool for tool in MEMORY_TOOLS if tool.name == "create_memory")
    command = ToolCommand(
        tool=tool.name, arguments={"title": "Title", "markdown": "Text"}
    )
    result = asyncio.run(controller.use_memory(tool, command, "alpha", False))
    assert result == {"error": "Storage unavailable"}
    controller.services.json_request.assert_awaited_once()
