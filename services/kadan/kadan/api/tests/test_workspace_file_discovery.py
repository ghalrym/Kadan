"""File discovery units with the storage and computer boundaries replaced."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from kadan.api.config import ServiceError
from kadan.runner.agent_runs import RunConversation
from kadan.runner.tools import TOOLS_BY_NAME
from kadan.runner.workspace_files import WorkspaceFiles


def asset(number=1, **changes):
    return {
        "id": str(UUID(int=number)),
        "agent_key": "ember",
        "title": "Weekly report",
        "filename": "report.pdf",
        "media_type": "application/pdf",
        "size_bytes": 42,
        "sha256": "a" * 64,
        "version": 2,
        "created_at": "2026-10-01T12:00:00Z",
        "updated_at": "2026-10-22T12:00:00Z",
        "deleted_at": None,
        "tags": ["weekly"],
        "sources": [{"label": "Source", "url": "https://example.com/report"}],
        **changes,
    }


def boundary(rows):
    return SimpleNamespace(json_request=AsyncMock(return_value=rows))


def conversation(storage):
    run = SimpleNamespace(id=uuid4(), agent_key="ember", transcript=[])
    return RunConversation(SimpleNamespace(services=storage, owned=AsyncMock()), run)


def invoke(name, storage, **arguments):
    current = conversation(storage)
    result = asyncio.run(
        TOOLS_BY_NAME[name].ainvoke(
            arguments, config={"configurable": {"conversation": current}}
        )
    )
    current.runs.owned.assert_awaited_once_with(current.run)
    assert current.end_run is False
    return result


@pytest.mark.parametrize(
    "name, arguments, query",
    [
        ("list_workspace_files", {}, ""),
        ("list_workspace_files", {"query": "  WEEKLY report  "}, "WEEKLY report"),
    ],
)
def test_discovery_queries_only_current_active_namespace(name, arguments, query):
    storage = boundary([asset()])
    result = invoke(name, storage, **arguments)
    storage.json_request.assert_awaited_once_with(
        "Tsuchi",
        "GET",
        "/api/agents/ember/assets",
        params={"q": query, "trash": "false", "limit": 21, "offset": 0},
    )
    assert result["tool"] == name
    assert result["next_offset"] is None
    assert result["files"] == [
        {
            "file_id": str(UUID(int=1)),
            "title": "Weekly report",
            "filename": "report.pdf",
            "media_type": "application/pdf",
            "size_bytes": 42,
            "sha256": "a" * 64,
            "version": 2,
            "created_at": "2026-10-01T12:00:00Z",
            "updated_at": "2026-10-22T12:00:00Z",
            "download_url": f"/api/agents/ember/assets/{UUID(int=1)}/content?download=true",
        }
    ]


def test_duplicate_names_keep_distinct_ids_and_storage_order():
    storage = boundary([asset(2, updated_at="2026-10-23T00:00:00Z"), asset(1)])
    result = invoke("list_workspace_files", storage)
    assert [row["filename"] for row in result["files"]] == ["report.pdf"] * 2
    assert [row["file_id"] for row in result["files"]] == [
        str(UUID(int=2)),
        str(UUID(int=1)),
    ]


def test_page_uses_one_lookahead_and_continues_at_next_matching_offset():
    storage = boundary([asset(number) for number in range(1, 52)])
    result = invoke("list_workspace_files", storage, query="report", limit=50, offset=7)
    assert len(result["files"]) == 50
    assert result["next_offset"] == 57
    assert storage.json_request.call_args.kwargs["params"] == {
        "q": "report",
        "trash": "false",
        "limit": 51,
        "offset": 7,
    }


def test_exact_full_final_page_does_not_offer_another_page():
    result = invoke("list_workspace_files", boundary([asset(1), asset(2)]), limit=2)
    assert len(result["files"]) == 2
    assert result["next_offset"] is None


@pytest.mark.parametrize(
    "name, arguments",
    [("list_workspace_files", {}), ("list_workspace_files", {"query": "absent"})],
)
def test_no_matches_is_an_empty_success(name, arguments):
    assert invoke(name, boundary([]), **arguments) == {
        "tool": name,
        "files": [],
        "next_offset": None,
    }


@pytest.mark.parametrize(
    "rows, message",
    [
        ({"files": []}, "invalid file list"),
        ([asset()] * 22, "invalid file list"),
        ([asset(id="not-a-uuid")], "invalid file metadata"),
        ([asset(agent_key="other")], "different Workspace file"),
        ([asset(deleted_at="2026-10-23T00:00:00Z")], "Trash"),
        ([asset(updated_at="yesterday")], "invalid file metadata"),
        ([asset(size_bytes="42")], "invalid file metadata"),
        ([asset(1), asset(1)], "duplicate file identities"),
        ([asset(1), asset(2, agent_key="other")], "different Workspace file"),
    ],
)
def test_invalid_responses_are_errors_without_partial_results(rows, message):
    result = invoke(
        "list_workspace_files", boundary(rows), limit=1 if len(rows) == 2 else 20
    )
    assert message in result["error"]
    assert "files" not in result
    assert "next_offset" not in result


def test_storage_failure_is_not_reported_as_no_matches():
    storage = boundary([])
    storage.json_request.side_effect = ServiceError("Tsuchi is unavailable", 503)
    result = invoke("list_workspace_files", storage, query="report")
    assert result == {
        "tool": "list_workspace_files",
        "error": "Tsuchi is unavailable",
    }


def test_cancelled_run_does_not_query_storage():
    storage = boundary([])
    current = conversation(storage)
    current.runs.owned.side_effect = ServiceError("Run cancelled", 409)
    with pytest.raises(ServiceError, match="Run cancelled"):
        asyncio.run(
            current.list_workspace_files(
                TOOLS_BY_NAME["list_workspace_files"].args_schema()
            )
        )
    storage.json_request.assert_not_awaited()


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("list_workspace_files", {"limit": 0}),
        ("list_workspace_files", {"limit": 51}),
        ("list_workspace_files", {"offset": -1}),
        ("list_workspace_files", {"agent_key": "other"}),
        ("list_workspace_files", {"trash": True}),
        ("list_workspace_files", {"order": "name"}),
        ("list_workspace_files", {"query": []}),
        ("list_workspace_files", {"query": "x" * 501}),
        (
            "list_workspace_files",
            {"query": "report", "media_type": "application/pdf"},
        ),
    ],
)
def test_discovery_schema_bounds_query_pages_and_scope(name, arguments):
    with pytest.raises(ValidationError):
        TOOLS_BY_NAME[name].args_schema.model_validate(arguments)


@pytest.mark.parametrize(
    "changes", [{"agent_key": "other"}, {"deleted_at": "2026-10-23T00:00:00Z"}]
)
def test_open_rechecks_selected_file_after_discovery(changes):
    storage = boundary([asset()])
    result = invoke("list_workspace_files", storage)
    file_id = UUID(result["files"][0]["file_id"])
    storage.json_request.return_value = asset(**changes)
    with pytest.raises(ServiceError):
        asyncio.run(
            WorkspaceFiles(storage, "ember").open(
                file_id, uuid4(), AsyncMock(), replayed=False
            )
        )
    assert storage.json_request.call_args.args == (
        "Tsuchi",
        "GET",
        f"/api/agents/ember/assets/{file_id}",
    )
