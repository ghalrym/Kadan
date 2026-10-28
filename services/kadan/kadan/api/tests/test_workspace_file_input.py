"""Workspace input copies with mocked HTTP and computer boundaries."""

import asyncio
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from kadan.api.config import ServiceError
from kadan.api.sandbox_client import MAX_FILE_BYTES, DownloadedFile
from kadan.runner.agent_runs import RunConversation
from kadan.runner.tools import TOOLS_BY_NAME
from kadan.runner.tools.workspace import OpenWorkspaceFileArguments
from kadan.runner.workspace_files import WorkspaceFiles

PAYLOAD = b"\x00\xffbinary\r\nfile"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


async def chunks(*values):
    for value in values:
        if isinstance(value, BaseException):
            raise value
        yield value


class Boundary:
    def __init__(self):
        self.file_id = uuid4()
        self.action_id = uuid4()
        self.path = f"/tmp/kadan-workspace-{self.action_id}-report.pdf"
        self.metadata = {
            "id": str(self.file_id),
            "agent_key": "ember",
            "title": "Report",
            "filename": "report.pdf",
            "media_type": "application/pdf",
            "size_bytes": len(PAYLOAD),
            "sha256": DIGEST,
            "version": 1,
            "deleted_at": None,
        }
        self.json_request = AsyncMock(side_effect=lambda *args: self.metadata)
        self.response = httpx.Response(200, content=chunks(PAYLOAD))
        self.requests = []
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self.respond))
        self.settings = SimpleNamespace(tsuchi_url="http://tsuchi")
        self.files = {}
        self.outputs = []
        self.sources = []
        self.sandbox = SimpleNamespace(
            download_file=AsyncMock(side_effect=self.download),
            upload_file=AsyncMock(side_effect=self.upload),
        )
        self.owned = AsyncMock()

    def respond(self, request):
        self.requests.append(request)
        return self.response

    async def download(self, path, output):
        self.outputs.append(output)
        if path not in self.files:
            raise ServiceError("Missing computer file", 404, "file_not_found")
        data = self.files[path]
        output.write(data)
        output.seek(0)
        return DownloadedFile(
            size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
        )

    async def upload(self, path, source):
        self.sources.append(source)
        self.files[path] = source.read()

    def open(self, replayed=False):
        return WorkspaceFiles(self, "ember").open(
            self.file_id, self.action_id, self.owned, replayed=replayed
        )


def test_open_copies_verified_bytes_to_a_command_unique_path_without_mutating_storage():
    boundary = Boundary()
    result = asyncio.run(boundary.open())
    assert result["opened"] is True
    assert result["path"] == boundary.path
    assert result["file_id"] == str(boundary.file_id)
    assert boundary.files == {boundary.path: PAYLOAD}
    assert boundary.owned.await_count == 2
    boundary.json_request.assert_awaited_once_with(
        "Tsuchi", "GET", f"/api/agents/ember/assets/{boundary.file_id}"
    )
    assert len(boundary.requests) == 1
    assert (
        boundary.requests[0].url
        == f"http://tsuchi/api/agents/ember/assets/{boundary.file_id}/content"
    )
    assert boundary.requests[0].headers["accept-encoding"] == "identity"
    assert boundary.sandbox.upload_file.await_count == 1
    assert boundary.sandbox.download_file.await_count == 2
    assert boundary.response.is_closed
    assert all(output.closed for output in [*boundary.outputs, *boundary.sources])


@pytest.mark.parametrize(
    "metadata",
    [
        {"agent_key": "other"},
        {"id": str(uuid4())},
        {"deleted_at": "2026-10-22T12:00:00Z"},
        {"sha256": "invalid"},
        {"size_bytes": MAX_FILE_BYTES + 1},
    ],
)
def test_invalid_wrong_agent_trashed_and_oversized_files_never_start_transfers(
    metadata,
):
    boundary = Boundary()
    boundary.metadata.update(metadata)
    with pytest.raises(ServiceError):
        asyncio.run(boundary.open())
    assert boundary.requests == []
    boundary.sandbox.upload_file.assert_not_awaited()
    boundary.sandbox.download_file.assert_not_awaited()


@pytest.mark.parametrize(
    "response",
    [
        lambda: httpx.Response(404, json={"detail": "missing"}),
        lambda: httpx.Response(206, content=chunks(PAYLOAD)),
        lambda: httpx.Response(200, content=chunks(PAYLOAD[:-1])),
        lambda: httpx.Response(200, content=chunks(PAYLOAD + b"extra")),
        lambda: httpx.Response(200, content=chunks(b"x" * len(PAYLOAD))),
        lambda: httpx.Response(
            200, content=chunks(PAYLOAD), headers={"content-length": "-1"}
        ),
        lambda: httpx.Response(
            200, content=chunks(PAYLOAD), headers={"content-length": "999"}
        ),
        lambda: httpx.Response(
            200, content=chunks(PAYLOAD), headers={"content-length": "9" * 5000}
        ),
        lambda: httpx.Response(
            200, content=chunks(PAYLOAD), headers={"content-encoding": "gzip"}
        ),
        lambda: httpx.Response(
            200, content=chunks(PAYLOAD[:2], httpx.ReadError("interrupted"))
        ),
    ],
)
def test_invalid_incomplete_or_changed_storage_bytes_never_reach_computer(response):
    boundary = Boundary()
    boundary.response = response()
    with pytest.raises(ServiceError):
        asyncio.run(boundary.open())
    assert boundary.response.is_closed
    boundary.sandbox.upload_file.assert_not_awaited()
    boundary.sandbox.download_file.assert_not_awaited()


def test_source_transfer_has_a_total_deadline(monkeypatch):
    monkeypatch.setattr("kadan.runner.workspace_files.OPERATION_SECONDS", 0.01)

    async def stalled():
        yield b"x"
        await asyncio.Event().wait()

    boundary = Boundary()
    boundary.response = httpx.Response(200, content=stalled())
    with pytest.raises(ServiceError, match="did not complete"):
        asyncio.run(boundary.open())
    assert boundary.response.is_closed
    boundary.sandbox.upload_file.assert_not_awaited()


def test_existing_edited_copy_is_not_overwritten():
    boundary = Boundary()
    boundary.files[boundary.path] = b"user edits"
    with pytest.raises(ServiceError, match="left unchanged"):
        asyncio.run(boundary.open())
    assert boundary.files[boundary.path] == b"user edits"
    boundary.sandbox.upload_file.assert_not_awaited()


def test_existing_identical_copy_is_reused_without_upload():
    boundary = Boundary()
    boundary.files[boundary.path] = PAYLOAD
    result = asyncio.run(boundary.open())
    assert result["opened"] is True
    boundary.sandbox.upload_file.assert_not_awaited()


def test_lost_upload_receipt_can_be_confirmed_by_exact_readback():
    boundary = Boundary()

    async def lost_receipt(path, source):
        await boundary.upload(path, source)
        raise ServiceError("Upload response lost", 503)

    boundary.sandbox.upload_file.side_effect = lost_receipt
    result = asyncio.run(boundary.open())
    assert result["opened"] is True
    assert boundary.sandbox.upload_file.await_count == 1


def test_unconfirmed_upload_is_never_repeated():
    boundary = Boundary()
    boundary.sandbox.upload_file.side_effect = ServiceError("Upload response lost", 503)
    with pytest.raises(ServiceError, match="unconfirmed") as raised:
        asyncio.run(boundary.open())
    assert boundary.path in str(raised.value)
    assert boundary.sandbox.upload_file.await_count == 1


def test_success_receipt_without_matching_readback_does_not_confirm_copy():
    boundary = Boundary()

    async def corrupt_copy(path, source):
        boundary.files[path] = b"changed during upload"

    boundary.sandbox.upload_file.side_effect = corrupt_copy
    with pytest.raises(ServiceError, match="unconfirmed"):
        asyncio.run(boundary.open())
    assert boundary.sandbox.upload_file.await_count == 1


def test_replay_recovers_copy_without_downloading_storage_or_uploading_again():
    boundary = Boundary()
    boundary.files[boundary.path] = PAYLOAD
    assert asyncio.run(boundary.open(replayed=True))["path"] == boundary.path
    assert boundary.requests == []
    boundary.sandbox.upload_file.assert_not_awaited()


@pytest.mark.parametrize("existing", [None, b"user edits"])
def test_replay_leaves_missing_or_edited_copy_unchanged(existing):
    boundary = Boundary()
    if existing is not None:
        boundary.files[boundary.path] = existing
    with pytest.raises(ServiceError, match="Nothing was uploaded again"):
        asyncio.run(boundary.open(replayed=True))
    assert boundary.files.get(boundary.path) == existing
    assert boundary.requests == []
    boundary.sandbox.upload_file.assert_not_awaited()


@pytest.mark.parametrize("during_transfer", [False, True])
def test_lost_ownership_prevents_computer_write(during_transfer):
    boundary = Boundary()
    boundary.owned.side_effect = (
        [None, ServiceError("Run cancelled", 409)]
        if during_transfer
        else ServiceError("Run cancelled", 409)
    )
    with pytest.raises(ServiceError, match="Run cancelled"):
        asyncio.run(boundary.open())
    boundary.sandbox.upload_file.assert_not_awaited()
    if not during_transfer:
        boundary.json_request.assert_not_awaited()
        assert boundary.requests == []


def test_task_cancellation_closes_source_stream_and_never_uploads():
    boundary = Boundary()
    boundary.response = httpx.Response(200, content=chunks(asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(boundary.open())
    assert boundary.response.is_closed
    boundary.sandbox.upload_file.assert_not_awaited()


def test_tool_dispatch_uses_current_agent_and_action_without_mutation_lock():
    boundary = Boundary()
    run = SimpleNamespace(id=uuid4(), agent_key="ember", transcript=[])
    conversation = RunConversation(
        SimpleNamespace(services=boundary, owned=boundary.owned), run
    )
    result = asyncio.run(
        TOOLS_BY_NAME["open_workspace_file"].ainvoke(
            {"file_id": str(boundary.file_id)},
            config={"configurable": {"conversation": conversation}},
        )
    )
    assert result["tool"] == "open_workspace_file"
    assert result["path"] == f"/tmp/kadan-workspace-{conversation.action_id}-report.pdf"
    assert conversation.end_run is False
    assert boundary.owned.await_args.args == (run,)


@pytest.mark.parametrize(
    "extra",
    [
        {"path": "/home/user/existing"},
        {"agent_key": "other"},
        {"action_id": str(uuid4())},
    ],
)
def test_open_tool_does_not_accept_destination_or_namespace_overrides(extra):
    with pytest.raises(ValidationError):
        OpenWorkspaceFileArguments(file_id=uuid4(), **extra)


def test_long_unicode_filename_stays_within_a_single_filesystem_name():
    boundary = Boundary()
    boundary.metadata["filename"] = "🌻" * 200 + ".pdf"
    result = asyncio.run(boundary.open())
    assert len(result["path"].rsplit("/", 1)[-1].encode("utf-8")) <= 255
    assert boundary.files[result["path"]] == PAYLOAD
