"""File publication units with computer and storage boundaries replaced."""

import asyncio
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4, uuid5

import pytest
from pydantic import ValidationError

from kadan.api.config import ServiceError
from kadan.runner.agent_runs import RunConversation
from kadan.runner.tools import TOOLS_BY_NAME
from kadan.runner.tools.workspace import (
    ReadWorkspaceFileArguments,
    SaveWorkspaceFileArguments,
)
from kadan.runner.workspace_files import WorkspaceFiles, file_name

PAYLOAD = b"\x00\xffbinary\r\nfile"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def asset(file_id, **changes):
    return {
        "id": str(file_id),
        "agent_key": "ember",
        "title": "Report",
        "filename": "report.pdf",
        "media_type": "application/pdf",
        "size_bytes": len(PAYLOAD),
        "sha256": DIGEST,
        "version": 1,
        "deleted_at": None,
        **changes,
    }


class Boundary:
    def __init__(self):
        self.calls = []
        self.outputs = []
        self.reply = None
        self.sandbox = SimpleNamespace(
            download_file=AsyncMock(side_effect=self.download)
        )
        self.owned = AsyncMock()

    async def download(self, path, output):
        self.outputs.append(output)
        output.write(PAYLOAD)
        output.seek(0)
        return SimpleNamespace(size_bytes=len(PAYLOAD), sha256=DIGEST)

    async def json_request(self, service, method, path, **kwargs):
        self.calls.append((service, method, path, kwargs))
        if method == "POST":
            filename, output, media_type = kwargs["files"]["file"]
            assert filename == "report.pdf"
            assert media_type == "application/octet-stream"
            assert output.read() == PAYLOAD
            value = asset(kwargs["data"]["asset_id"])
        else:
            value = asset(path.rsplit("/", 1)[-1])
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply if self.reply is not None else value

    def save(self, *, replayed=False, file_id=None):
        return WorkspaceFiles(self, "ember").save(
            file_id or uuid4(),
            "/home/user/report.pdf",
            "Report",
            self.owned,
            replayed=replayed,
        )


def test_save_streams_exact_bytes_to_its_agent_and_returns_a_download_link():
    boundary = Boundary()
    file_id = uuid4()
    result = asyncio.run(boundary.save(file_id=file_id))
    service, method, path, kwargs = boundary.calls[0]
    assert (service, method, path) == ("Tsuchi", "POST", "/api/agents/ember/assets")
    assert kwargs["data"] == {"title": "Report", "asset_id": str(file_id)}
    assert result == {
        "saved": True,
        "file_id": str(file_id),
        "title": "Report",
        "filename": "report.pdf",
        "media_type": "application/pdf",
        "size_bytes": len(PAYLOAD),
        "sha256": DIGEST,
        "download_url": f"/api/agents/ember/assets/{file_id}/content?download=true",
    }
    assert boundary.owned.await_count == 2
    assert boundary.outputs[0].closed


@pytest.mark.parametrize(
    "changes",
    [
        {"id": str(uuid4())},
        {"agent_key": "other"},
        {"filename": "other.pdf"},
        {"size_bytes": len(PAYLOAD) - 1},
        {"sha256": "0" * 64},
        {"deleted_at": "2026-10-22T12:00:00Z"},
        {"sha256": "invalid"},
    ],
)
def test_mismatched_or_trashed_receipt_never_confirms_publication(changes):
    boundary = Boundary()
    file_id = uuid4()
    boundary.reply = asset(file_id, **changes)
    with pytest.raises(ServiceError, match="unconfirmed"):
        asyncio.run(boundary.save(file_id=file_id))
    assert len(boundary.calls) == 1
    assert boundary.outputs[0].closed


@pytest.mark.parametrize(
    "failure", [ServiceError("Storage unavailable", 503), TimeoutError()]
)
def test_uncertain_upload_is_not_repeated_and_tempfile_is_closed(failure):
    boundary = Boundary()
    boundary.reply = failure
    with pytest.raises(ServiceError, match="read_workspace_file"):
        asyncio.run(boundary.save())
    assert len(boundary.calls) == 1
    assert boundary.outputs[0].closed


def test_restart_recovers_original_saved_file_without_reading_changed_source():
    boundary = Boundary()
    file_id = uuid4()
    result = asyncio.run(boundary.save(replayed=True, file_id=file_id))
    assert result["saved"] is True
    assert result["file_id"] == str(file_id)
    assert [call[1] for call in boundary.calls] == ["GET"]
    boundary.sandbox.download_file.assert_not_awaited()


@pytest.mark.parametrize(
    "reply", [ServiceError("Not found", 404), ServiceError("Storage down", 503)]
)
def test_restart_without_confirmation_never_uploads_source_again(reply):
    boundary = Boundary()
    boundary.reply = reply
    with pytest.raises(
        ServiceError, match="source has not been read or uploaded again"
    ):
        asyncio.run(boundary.save(replayed=True))
    assert [call[1] for call in boundary.calls] == ["GET"]
    boundary.sandbox.download_file.assert_not_awaited()


def test_explicit_read_checks_an_uncertain_save_and_does_not_read_contents():
    boundary = Boundary()
    file_id = uuid4()
    result = asyncio.run(WorkspaceFiles(boundary, "ember").read(file_id))
    assert result["file_id"] == str(file_id)
    assert "saved" not in result
    assert boundary.calls[0][2] == f"/api/agents/ember/assets/{file_id}"
    boundary.sandbox.download_file.assert_not_awaited()


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


def test_cancelled_response_cannot_start_a_download():
    boundary = Boundary()
    boundary.owned.side_effect = ServiceError("Run cancelled", 409)
    with pytest.raises(ServiceError, match="Run cancelled"):
        asyncio.run(boundary.save())
    boundary.sandbox.download_file.assert_not_awaited()
    assert boundary.calls == []


def test_cancellation_during_download_prevents_upload():
    boundary = Boundary()
    boundary.owned.side_effect = [None, ServiceError("Run cancelled", 409)]
    with pytest.raises(ServiceError, match="Run cancelled"):
        asyncio.run(boundary.save())
    assert boundary.calls == []
    assert boundary.outputs[0].closed


def test_failed_download_never_uploads_partial_output():
    boundary = Boundary()

    async def fail(path, output):
        boundary.outputs.append(output)
        output.write(b"partial")
        raise ServiceError("Source changed", 409)

    boundary.sandbox.download_file.side_effect = fail
    with pytest.raises(ServiceError, match="Source changed"):
        asyncio.run(boundary.save())
    assert boundary.calls == []
    assert boundary.outputs[0].closed


def test_task_cancellation_propagates_and_closes_tempfile():
    boundary = Boundary()
    boundary.reply = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(boundary.save())
    assert boundary.outputs[0].closed


def test_conversation_uses_durable_action_identity_and_does_not_deliver_a_message():
    boundary = Boundary()
    run = SimpleNamespace(id=uuid4(), agent_key="ember", transcript=["saved command"])
    runs = SimpleNamespace(services=boundary, owned=boundary.owned)
    conversation = RunConversation(runs, run)
    result = asyncio.run(
        TOOLS_BY_NAME["save_workspace_file"].ainvoke(
            {"path": "/home/user/report.pdf", "title": "Report"},
            config={"configurable": {"conversation": conversation}},
        )
    )
    assert result["tool"] == "save_workspace_file"
    assert result["file_id"] == str(uuid5(run.id, "command:1"))
    assert conversation.end_run is False
    assert len(boundary.calls) == 1
    assert boundary.calls[0][2].endswith("/assets")
    assert boundary.owned.await_args.args == (run,)


def test_conversation_reports_uncertain_file_id_for_reconciliation():
    boundary = Boundary()
    boundary.reply = ServiceError("Upload response lost", 503)
    run = SimpleNamespace(id=uuid4(), agent_key="ember", transcript=[])
    conversation = RunConversation(
        SimpleNamespace(services=boundary, owned=boundary.owned), run
    )
    result = asyncio.run(
        conversation.save_workspace_file(SaveWorkspaceFileArguments(path="/report.pdf"))
    )
    assert result["file_id"] == str(conversation.action_id)
    assert "saved" not in result
    assert "read_workspace_file" in result["error"]


@pytest.mark.parametrize(
    "arguments",
    [
        {"path": "relative.pdf"},
        {"path": "/bad\0name"},
        {"path": "/file", "title": " "},
        {"path": "/file", "agent_key": "other"},
        {"path": "/file", "file_id": str(uuid4())},
    ],
)
def test_file_save_schema_rejects_invalid_path_title_and_destination_override(
    arguments,
):
    with pytest.raises(ValidationError):
        SaveWorkspaceFileArguments.model_validate(arguments)


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/home/user/report.pdf", "report.pdf"),
        ("/tmp/ padded.txt ", "padded.txt"),
        ("/tmp/line\nbreak.txt", "linebreak.txt"),
        ("/tmp/..", "attachment"),
    ],
)
def test_filename_matches_tsuchi_display_name_rules(path, expected):
    assert file_name(path) == expected


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


def test_local_staging_failure_is_a_tool_error_without_upload():
    boundary = Boundary()
    boundary.sandbox.download_file.side_effect = OSError("Temporary disk full")
    run = SimpleNamespace(id=uuid4(), agent_key="ember", transcript=[])
    conversation = RunConversation(
        SimpleNamespace(services=boundary, owned=boundary.owned), run
    )
    result = asyncio.run(
        conversation.save_workspace_file(SaveWorkspaceFileArguments(path="/report.pdf"))
    )
    assert "Temporary disk full" in result["error"]
    assert "saved" not in result
    assert boundary.calls == []


def test_unconfigured_computer_does_not_create_a_file():
    boundary = Boundary()
    boundary.sandbox.download_file.side_effect = ServiceError(
        "The computer is not configured", 503
    )
    with pytest.raises(ServiceError, match="not configured"):
        asyncio.run(boundary.save())
    assert boundary.calls == []


def test_quoted_source_filename_does_not_make_a_saved_upload_look_unconfirmed():
    boundary = Boundary()
    file_id = uuid4()
    boundary.json_request = AsyncMock(
        return_value=asset(file_id, filename="report_final.pdf")
    )
    result = asyncio.run(
        WorkspaceFiles(boundary, "ember").save(
            file_id,
            '/home/user/report"final.pdf',
            None,
            boundary.owned,
            replayed=False,
        )
    )
    assert result["saved"] is True
    assert result["filename"] == "report_final.pdf"
    assert (
        boundary.json_request.call_args.kwargs["files"]["file"][0] == "report_final.pdf"
    )
