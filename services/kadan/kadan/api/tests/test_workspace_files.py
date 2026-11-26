"""Workspace file metadata reads with the storage boundary replaced."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from kadan.api.config import ServiceError
from kadan.runner.agent_runs import RunConversation
from kadan.runner.tools import TOOLS_BY_NAME
from kadan.runner.tools.workspace import ReadWorkspaceFileArguments
from kadan.runner.workspace_files import WorkspaceFiles


def asset(file_id, **changes):
    return {
        "id": str(file_id),
        "agent_key": "ember",
        "title": "Report",
        "filename": "report.pdf",
        "media_type": "application/pdf",
        "size_bytes": 16,
        "sha256": "0" * 64,
        "version": 1,
        "deleted_at": None,
        **changes,
    }


class Boundary:
    def __init__(self):
        self.calls = []
        self.reply = None
        self.owned = AsyncMock()

    async def json_request(self, service, method, path, **kwargs):
        self.calls.append((service, method, path, kwargs))
        return self.reply if self.reply is not None else asset(path.rsplit("/", 1)[-1])


def test_read_returns_metadata_and_a_download_link_without_contents():
    boundary = Boundary()
    file_id = uuid4()
    result = asyncio.run(WorkspaceFiles(boundary, "ember").read(file_id))
    assert result["file_id"] == str(file_id)
    assert result["download_url"] == (
        f"/api/agents/ember/assets/{file_id}/content?download=true"
    )
    assert boundary.calls == [
        ("Tsuchi", "GET", f"/api/agents/ember/assets/{file_id}", {})
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"agent_key": "other"},
        {"id": str(uuid4())},
        {"deleted_at": "2026-10-22T12:00:00Z"},
    ],
)
def test_read_rejects_wrong_namespace_identity_and_trash(changes):
    boundary = Boundary()
    file_id = uuid4()
    boundary.reply = asset(file_id, **changes)
    with pytest.raises(ServiceError):
        asyncio.run(WorkspaceFiles(boundary, "ember").read(file_id))


def test_read_tool_dispatches_in_current_conversation():
    boundary = Boundary()
    run = SimpleNamespace(id=uuid4(), agent_key="ember", transcript=[])
    conversation = RunConversation(
        SimpleNamespace(services=boundary, owned=boundary.owned), run
    )
    file_id = uuid4()
    result = asyncio.run(
        TOOLS_BY_NAME["read_workspace_file"].ainvoke(
            ReadWorkspaceFileArguments(file_id=file_id).model_dump(),
            config={"configurable": {"conversation": conversation}},
        )
    )
    assert result["file_id"] == str(file_id)
    assert result["tool"] == "read_workspace_file"
