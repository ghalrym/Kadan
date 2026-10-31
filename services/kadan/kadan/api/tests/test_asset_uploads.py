"""Browser uploads with local spooled files and a recording storage boundary."""

import asyncio
import hashlib
from io import BytesIO
from tempfile import SpooledTemporaryFile
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import UploadFile

from kadan.api.asset_uploads import upload_asset
from kadan.api.config import ServiceError
from kadan.api.routes.agents.assets import upload_asset as upload_route

PAYLOAD = b"\x00binary\xff\r\n"
NOW = "2026-10-22T12:00:00Z"


class Storage:
    def __init__(self, changes=None, failure=None):
        self.calls = []
        self.changes = changes or {}
        self.failure = failure

    async def json_request(self, service, method, path, **kwargs):
        name, source, media_type = kwargs["files"]["file"]
        self.calls.append((service, method, path, kwargs, source.read()))
        assert media_type == "application/octet-stream"
        if self.failure:
            raise self.failure
        return {
            "id": kwargs["data"]["asset_id"],
            "agent_key": "ember",
            "title": kwargs["data"]["title"],
            "filename": name,
            "media_type": "application/octet-stream",
            "is_image": False,
            "size_bytes": len(PAYLOAD),
            "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
            "version": 1,
            "created_at": NOW,
            "updated_at": NOW,
            "deleted_at": None,
            **self.changes,
        }


def upload(data=PAYLOAD, **kwargs):
    return UploadFile(BytesIO(data), filename='report"final.bin', **kwargs)


def test_upload_streams_bytes_with_the_stable_id_and_confirms_hash():
    storage = Storage()
    source = upload()
    asset_id = uuid4()
    result = asyncio.run(upload_asset(storage, "ember", asset_id, source, None))
    assert result.id == asset_id
    assert result.filename == "report_final.bin"
    assert result.title == "report_final.bin"
    assert storage.calls[0][:3] == ("Tsuchi", "POST", "/api/agents/ember/assets")
    assert storage.calls[0][-1] == PAYLOAD
    assert storage.calls[0][3]["data"]["asset_id"] == str(asset_id)
    assert not source.file.closed  # The caller owns the spool.
    source.file.close()


def test_retry_reuses_the_exact_upload_identity_and_metadata():
    storage = Storage()
    asset_id = uuid4()
    for _ in range(2):
        source = upload()
        asyncio.run(upload_asset(storage, "ember", asset_id, source, "My file"))
        source.file.close()
    assert storage.calls[0][3]["data"] == storage.calls[1][3]["data"]
    assert storage.calls[0][-1] == storage.calls[1][-1]


@pytest.mark.parametrize("reported_size", [None, 999999])
def test_upload_cap_rejects_reported_and_actual_oversize_before_storage(
    monkeypatch, reported_size
):
    monkeypatch.setattr("kadan.api.asset_uploads.MAX_UPLOAD_BYTES", 3)
    storage = Storage()
    source = upload(size=reported_size)
    with pytest.raises(ServiceError) as error:
        asyncio.run(upload_asset(storage, "ember", uuid4(), source, None))
    assert error.value.status == 413
    assert storage.calls == []
    source.file.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"id": str(uuid4())},
        {"agent_key": "another"},
        {"size_bytes": 0},
        {"sha256": "0" * 64},
        {"sha256": "invalid"},
        {"filename": "wrong.bin"},
        {"deleted_at": NOW},
    ],
)
def test_invalid_receipts_do_not_report_a_successful_upload(changes):
    source = upload()
    storage = Storage(changes=changes)
    with pytest.raises(ServiceError, match="confirm"):
        asyncio.run(upload_asset(storage, "ember", uuid4(), source, None))
    assert len(storage.calls) == 1
    source.file.close()


@pytest.mark.parametrize(
    "failure",
    [
        ServiceError("Conflict", 409),
        ServiceError("Storage offline", 503),
        TimeoutError(),
    ],
)
def test_failed_upload_is_never_retried_implicitly(failure):
    storage = Storage(failure=failure)
    source = upload()
    with pytest.raises(ServiceError):
        asyncio.run(upload_asset(storage, "ember", uuid4(), source, None))
    assert len(storage.calls) == 1
    source.file.close()


@pytest.mark.parametrize(
    "failure", [None, ServiceError("Storage offline", 503), asyncio.CancelledError()]
)
def test_route_closes_the_spooled_source_on_success_error_and_cancellation(failure):
    storage = Storage(failure=failure)
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(services=storage))
    )
    source = upload()
    call = upload_route("ember", request, uuid4(), source, None)
    if failure is not None:
        with pytest.raises(type(failure)):
            asyncio.run(call)
    else:
        assert asyncio.run(call).filename == "report_final.bin"
    assert source.file.closed


def test_source_io_failure_does_not_contact_storage():
    source = upload()
    source.read = AsyncMock(side_effect=OSError("Disk read failed"))
    storage = Storage()
    with pytest.raises(ServiceError, match="could not read"):
        asyncio.run(upload_asset(storage, "ember", uuid4(), source, None))
    assert storage.calls == []
    source.file.close()


@pytest.mark.parametrize(
    "failure", [None, ServiceError("Storage offline", 503), asyncio.CancelledError()]
)
def test_rolled_spool_uses_real_async_io_and_is_closed_by_route(failure):
    with SpooledTemporaryFile(max_size=1, mode="w+b") as spool:
        spool.write(PAYLOAD)
        assert spool._rolled
        source = UploadFile(spool, filename='report"final.bin', size=len(PAYLOAD))
        storage = Storage(failure=failure)
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(services=storage))
        )
        call = upload_route("ember", request, uuid4(), source, None)
        if failure is None:
            assert asyncio.run(call).size_bytes == len(PAYLOAD)
        else:
            with pytest.raises(type(failure)):
                asyncio.run(call)
        assert spool.closed
        assert storage.calls[0][-1] == PAYLOAD
