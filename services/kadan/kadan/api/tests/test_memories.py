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
    ForgetMemoryArguments,
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


def call(handler, action, check_ownership=None):
    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = LocalServices(client, Settings(_env_file=None))
            return await action(
                AgentMemories(service, "alpha", check_ownership or AsyncMock())
            )

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
@pytest.mark.parametrize("operation", ["read", "update", "forget"])
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
                else memories.forget(MEMORY_ID, 3)
                if operation == "forget"
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
    controller.owned = AsyncMock(return_value=run)
    state = SimpleNamespace(
        run=run, command=ToolCommand(tool=tool.name, arguments={}), replayed=True
    )
    if tool.name == "list_memories":
        controller.services.json_request.return_value = [memory()]
    if tool.name == "read_memory":
        state.command.arguments = {"memory_id": str(MEMORY_ID)}
    asyncio.run(controller.invoke_tool(state))
    controller.owned.assert_awaited_once_with(run)
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
    controller.owned = AsyncMock()
    run = SimpleNamespace(agent_key="alpha")
    result = asyncio.run(controller.use_memory(tool, command, run, False))
    assert result == {"error": "Storage unavailable"}
    controller.services.json_request.assert_awaited_once()


def test_forget_trashes_the_read_version_and_returns_confirmation_without_content():
    calls = []

    def storage(request):
        calls.append(request.method)
        assert request.url.path == f"/api/agents/alpha/resources/{MEMORY_ID}"
        if request.method == "GET":
            return httpx.Response(200, json=memory())
        assert request.method == "DELETE"
        assert dict(request.url.params) == {"expected_version": "3"}
        assert request.content == b""
        return httpx.Response(200, json=memory(version=4, deleted_at=NOW))

    result = call(storage, lambda memories: memories.forget(MEMORY_ID, 3))

    assert calls == ["GET", "DELETE"]
    assert result == {
        "id": str(MEMORY_ID),
        "title": "Preferences",
        "version": 4,
        "deleted_at": NOW.replace("+00:00", "Z"),
    }


def test_stale_forget_does_not_write_or_refresh_the_expected_version():
    calls = []
    ownership = AsyncMock()

    def storage(request):
        calls.append(request.method)
        return httpx.Response(200, json=memory())

    with pytest.raises(ServiceError) as error:
        call(storage, lambda memories: memories.forget(MEMORY_ID, 2), ownership)

    assert error.value.code == "version_conflict"
    assert calls == ["GET"]
    ownership.assert_not_awaited()


@pytest.mark.parametrize("failure", ["conflict", "timeout"])
def test_failed_forget_is_not_retried(failure):
    calls = []

    def storage(request):
        calls.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, json=memory())
        if failure == "timeout":
            raise httpx.ReadTimeout("No response", request=request)
        return httpx.Response(409, json={"detail": "Memory changed"})

    with pytest.raises(ServiceError):
        call(storage, lambda memories: memories.forget(MEMORY_ID, 3))

    assert calls == ["GET", "DELETE"]


@pytest.mark.parametrize(
    "changes",
    [
        {"id": str(uuid4())},
        {"agent_key": "beta"},
        {"kind": "persona"},
        {"deleted_at": None},
        {"version": 3},
        {"version": 5},
    ],
)
def test_forget_does_not_confirm_an_invalid_storage_result(changes):
    def storage(request):
        result = (
            memory()
            if request.method == "GET"
            else memory(**{"version": 4, "deleted_at": NOW, **changes})
        )
        return httpx.Response(200, json=result)

    with pytest.raises(ServiceError):
        call(storage, lambda memories: memories.forget(MEMORY_ID, 3))


@pytest.mark.parametrize("operation", ["read", "forget"])
def test_memory_id_mismatch_is_rejected_before_any_mutation(operation):
    calls = []

    def storage(request):
        calls.append(request.method)
        return httpx.Response(200, json=memory(id=str(uuid4())))

    with pytest.raises(ServiceError, match="wrong memory"):
        call(
            storage,
            lambda memories: (
                memories.read(MEMORY_ID)
                if operation == "read"
                else memories.forget(MEMORY_ID, 3)
            ),
        )

    assert calls == ["GET"]


@pytest.mark.parametrize("deleted", [False, True])
def test_explicit_trash_read_reconciles_state_without_recalling_forgotten_content(
    deleted,
):
    def storage(request):
        assert request.method == "GET"
        return httpx.Response(200, json=memory(deleted_at=NOW if deleted else None))

    result = call(
        storage, lambda memories: memories.read(MEMORY_ID, include_trashed=True)
    )

    assert bool(result["deleted_at"]) == deleted
    assert result["version"] == 3
    assert ("markdown" in result) is not deleted
    assert ("tags" in result) is not deleted


@pytest.mark.parametrize("changes", [{"agent_key": "beta"}, {"kind": "space"}])
def test_trash_read_still_checks_namespace_and_kind(changes):
    def storage(request):
        return httpx.Response(200, json=memory(deleted_at=NOW, **changes))

    with pytest.raises(ServiceError):
        call(storage, lambda memories: memories.read(MEMORY_ID, include_trashed=True))


@pytest.mark.parametrize("operation", ["create", "update", "forget"])
def test_memory_writes_check_ownership_immediately_before_mutation(operation):
    events = []

    async def check():
        events.append("ownership")

    def storage(request):
        events.append(request.method)
        return httpx.Response(
            200,
            json=memory(
                version=4 if request.method != "GET" else 3,
                deleted_at=NOW if request.method == "DELETE" else None,
            ),
        )

    actions = {
        "create": lambda memories: memories.create("Title", "Text", []),
        "update": lambda memories: memories.update(MEMORY_ID, 3, "Title", "Text", None),
        "forget": lambda memories: memories.forget(MEMORY_ID, 3),
    }
    call(storage, actions[operation], check)

    assert (
        events
        == {
            "create": ["ownership", "POST"],
            "update": ["GET", "ownership", "PUT"],
            "forget": ["GET", "ownership", "DELETE"],
        }[operation]
    )


@pytest.mark.parametrize("operation", ["create", "update", "forget"])
def test_lost_run_ownership_prevents_memory_mutation(operation):
    calls = []

    def storage(request):
        calls.append(request.method)
        return httpx.Response(200, json=memory())

    actions = {
        "create": lambda memories: memories.create("Title", "Text", []),
        "update": lambda memories: memories.update(MEMORY_ID, 3, "Title", "Text", None),
        "forget": lambda memories: memories.forget(MEMORY_ID, 3),
    }
    ownership = AsyncMock(
        side_effect=ServiceError("Run was cancelled", 409, "lease_lost")
    )
    with pytest.raises(ServiceError, match="Run was cancelled"):
        call(storage, actions[operation], ownership)

    assert calls == ([] if operation == "create" else ["GET"])


def test_runner_supplies_current_run_to_memory_ownership_check():
    controller = object.__new__(AgentRuns)
    controller.services = SimpleNamespace(
        json_request=AsyncMock(
            side_effect=[memory(), memory(version=4, deleted_at=NOW)]
        )
    )
    controller.owned = AsyncMock()
    run = SimpleNamespace(agent_key="alpha")
    tool = next(tool for tool in MEMORY_TOOLS if tool.name == "forget_memory")
    command = ToolCommand(
        tool=tool.name, arguments={"memory_id": str(MEMORY_ID), "expected_version": 3}
    )

    result = asyncio.run(controller.use_memory(tool, command, run, False))

    assert result["deleted_at"]
    controller.owned.assert_awaited_once_with(run)
    assert controller.services.json_request.call_args.args[1] == "DELETE"


@pytest.mark.parametrize(
    "arguments",
    [
        {"memory_id": str(MEMORY_ID)},
        {"memory_id": str(MEMORY_ID), "expected_version": True},
        {"memory_id": str(MEMORY_ID), "expected_version": 0},
        {"memory_id": str(MEMORY_ID), "expected_version": 3, "agent_key": "beta"},
    ],
)
def test_forget_schema_rejects_invalid_versions_and_scope_override(arguments):
    with pytest.raises(ValidationError):
        ForgetMemoryArguments.model_validate(arguments)


def test_trash_read_option_requires_a_boolean():
    with pytest.raises(ValidationError):
        ReadMemoryArguments(memory_id=MEMORY_ID, include_trashed="true")
