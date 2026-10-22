"""Sandbox client tests against an in-memory HTTP transport, never a real computer."""

import asyncio
import hashlib
import json
from io import BytesIO
from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr

from kadan.api.config import ServiceError
from kadan.api.sandbox_client import (
    BROWSER_RECOVERED,
    MAX_RESULT_CHARACTERS,
    SandboxClient,
    segment,
)


def sandbox(handler, api_key: str = "secret"):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = SimpleNamespace(
        sandbox_url="http://kadan-sandbox:8080",
        sandbox_api_key=SecretStr(api_key),
    )
    return SandboxClient(SimpleNamespace(client=client, settings=settings))


def serving(*responses):
    """Record and answer each request in turn."""
    seen = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return queue.pop(0)

    return handler, seen


def test_run_sends_json_body_with_the_api_key_and_returns_json():
    handler, seen = serving(httpx.Response(200, json={"data": {"output": "hi"}}))
    result = asyncio.run(
        sandbox(handler).run("POST", "/v1/shell/exec", json_body={"command": "echo hi"})
    )
    assert result == {
        "status": 200,
        "content_type": "application/json",
        "body": {"data": {"output": "hi"}},
    }
    assert seen[0].method == "POST"
    assert seen[0].url == "http://kadan-sandbox:8080/v1/shell/exec"
    assert seen[0].headers["authorization"] == "Bearer secret"
    assert json.loads(seen[0].content) == {"command": "echo hi"}


def test_path_segments_are_encoded_so_they_stay_inside_the_route():
    handler, seen = serving(httpx.Response(200, json={}))
    asyncio.run(
        sandbox(handler).run("DELETE", f"/v1/browser/tabs/{segment('../../x')}")
    )
    assert seen[0].url.raw_path == b"/v1/browser/tabs/..%2F..%2Fx"


def test_query_parameters_and_binary_results():
    handler, seen = serving(
        httpx.Response(
            200, content=b"\xff\xd8jpeg", headers={"content-type": "image/jpeg"}
        )
    )
    result = asyncio.run(
        sandbox(handler).run("GET", "/v1/browser/screenshot", params={"format": "jpeg"})
    )
    assert seen[0].url.params["format"] == "jpeg"
    assert result == {
        "status": 200,
        "content_type": "image/jpeg",
        "bytes": 6,
        "note": "Binary content is not shown to the agent.",
    }


def test_multipart_uploads_send_files_and_fields():
    handler, seen = serving(httpx.Response(200, json={"success": True}))
    asyncio.run(
        sandbox(handler).run(
            "POST",
            "/v1/file/upload",
            data={"path": "/home/gem/a.txt"},
            files={"file": ("file", b"hello")},
        )
    )
    body = seen[0].content
    assert seen[0].headers["content-type"].startswith("multipart/form-data")
    assert b'name="file"; filename="file"' in body and b"hello" in body
    assert b'name="path"' in body and b"/home/gem/a.txt" in body


def test_large_results_are_truncated_for_the_transcript():
    handler, _ = serving(httpx.Response(200, json={"output": "x" * 10000}))
    result = asyncio.run(sandbox(handler).run("POST", "/v1/shell/exec", json_body={}))
    assert result["truncated"] is True
    assert len(result["body"]) == MAX_RESULT_CHARACTERS
    assert result["characters"] > MAX_RESULT_CHARACTERS


def test_sandbox_errors_return_their_status_for_the_agent_to_correct():
    handler, _ = serving(httpx.Response(422, json={"detail": "command required"}))
    result = asyncio.run(sandbox(handler).run("POST", "/v1/shell/exec", json_body={}))
    assert result["status"] == 422
    assert result["body"] == {"detail": "command required"}


def test_unreachable_computer_is_a_service_error():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ServiceError) as error:
        asyncio.run(sandbox(handler).context())
    assert "unavailable" in str(error.value)
    with pytest.raises(ServiceError) as error:
        asyncio.run(sandbox(handler).run("GET", "/v1/browser/tabs"))
    assert error.value.code == "computer_unavailable"


def test_requests_use_default_api_key_when_empty():
    handler, seen = serving(httpx.Response(200, json={}))
    asyncio.run(sandbox(handler, api_key="").run("GET", "/v1/browser/tabs"))
    assert seen[0].headers["authorization"] == "Bearer insecure-api-key"


def test_vnc_relay_targets_websockify_on_the_sandbox():
    assert (
        sandbox(lambda request: None).vnc_url() == "ws://kadan-sandbox:8080/websockify"
    )


def test_event_streams_return_what_arrived_within_the_window(monkeypatch):
    monkeypatch.setattr("kadan.api.sandbox_client.STREAM_SECONDS", 0.05)

    async def events():
        yield b"data: first\n\n"
        await asyncio.sleep(10)

    handler, _ = serving(
        httpx.Response(
            200, content=events(), headers={"content-type": "text/event-stream"}
        )
    )
    result = asyncio.run(sandbox(handler).run("GET", "/v1/file/watch/w1/events"))
    assert result["body"] == "data: first\n\n"
    assert result["incomplete"] is True


GATEWAY_TIMEOUT = httpx.Response(
    504, text="<html>504 Gateway Time-out</html>", headers={"content-type": "text/html"}
)


def routes(seen):
    return [(request.method, request.url.path) for request in seen]


@pytest.mark.parametrize(
    "path",
    ["/v1/browser/tabs", "/v1/browser/page/markdown", "/v1/browser/screenshot"],
)
def test_jammed_browser_read_is_restarted_and_retried_once(path):
    handler, seen = serving(
        GATEWAY_TIMEOUT,
        httpx.Response(200, json={"success": True}),
        httpx.Response(200, json={"data": [{"index": 0}]}),
    )
    result = asyncio.run(sandbox(handler).run("GET", path))
    assert routes(seen) == [
        ("GET", path),
        ("POST", "/v1/browser/restart"),
        ("GET", path),
    ]
    assert result["status"] == 200
    assert result["body"] == {"data": [{"index": 0}]}
    assert result["recovered"] == BROWSER_RECOVERED


@pytest.mark.parametrize(
    ("method", "path", "options"),
    [
        (
            "POST",
            "/v1/browser/actions",
            {"json_body": {"action_type": "CLICK", "x": 10, "y": 20}},
        ),
        (
            "POST",
            "/v1/browser/page/navigate",
            {"json_body": {"url": "https://example.com"}},
        ),
        (
            "GET",
            "/v1/browser/page/console",
            {"params": {"clear": True}},
        ),
    ],
)
def test_unconfirmed_browser_operation_preserves_state_without_replay(
    method, path, options
):
    handler, seen = serving(GATEWAY_TIMEOUT)
    result = asyncio.run(sandbox(handler).run(method, path, **options))
    assert routes(seen) == [(method, path)]
    assert result["status"] == 504
    assert "may or may not have run" in result["error"]
    assert "recovered" not in result


def test_browser_read_that_times_out_again_is_not_retried_twice():
    handler, seen = serving(
        GATEWAY_TIMEOUT,
        httpx.Response(200, json={"success": True}),
        GATEWAY_TIMEOUT,
    )
    result = asyncio.run(sandbox(handler).run("GET", "/v1/browser/page/markdown"))
    assert routes(seen) == [
        ("GET", "/v1/browser/page/markdown"),
        ("POST", "/v1/browser/restart"),
        ("GET", "/v1/browser/page/markdown"),
    ]
    assert result["status"] == 504


def test_failed_browser_restart_is_reported_without_a_retry():
    handler, seen = serving(GATEWAY_TIMEOUT, httpx.Response(500, json={}))
    result = asyncio.run(sandbox(handler).run("GET", "/v1/browser/tabs"))
    assert routes(seen) == [
        ("GET", "/v1/browser/tabs"),
        ("POST", "/v1/browser/restart"),
    ]
    assert result["status"] == 504
    assert (
        result["error"] == "The browser stopped responding and could not be restarted."
    )


@pytest.mark.parametrize(
    ("method", "path"),
    [("POST", "/v1/shell/exec"), ("POST", "/v1/browser/restart")],
)
def test_only_jammed_browser_calls_trigger_a_restart(method, path):
    handler, seen = serving(GATEWAY_TIMEOUT)
    result = asyncio.run(sandbox(handler).run(method, path, json_body={}))
    assert len(seen) == 1
    assert result["status"] == 504
    assert "recovered" not in result


class DownloadStream(httpx.AsyncByteStream):
    def __init__(self, chunks=(), error=None, *, wait=False):
        self.chunks = chunks
        self.error = error
        self.wait = wait
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.error:
            raise self.error
        if self.wait:
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("length_header", [True, False])
def test_download_preserves_binary_bytes_and_checksums_beyond_transcript_limit(
    length_header,
):
    content = bytes(range(256)) * 5000
    stream = DownloadStream([content[:70000], content[70000:]])
    headers = {"content-length": str(len(content))} if length_header else {}
    handler, seen = serving(httpx.Response(200, stream=stream, headers=headers))
    output = BytesIO()

    result = asyncio.run(
        sandbox(handler).download_file("/tmp/report & draft.bin", output)
    )

    assert result.size_bytes == len(content)
    assert result.sha256 == hashlib.sha256(content).hexdigest()
    assert output.tell() == 0
    assert output.read() == content
    assert not output.closed
    assert stream.closed
    assert len(seen) == 1
    assert seen[0].method == "GET"
    assert seen[0].url.path == "/v1/file/download"
    assert dict(seen[0].url.params) == {
        "path": "/tmp/report & draft.bin",
        "change_policy": "abort",
    }
    assert seen[0].headers["authorization"] == "Bearer secret"
    assert seen[0].headers["accept-encoding"] == "identity"


def test_download_accepts_an_empty_file():
    handler, _ = serving(httpx.Response(200, content=b""))
    output = BytesIO()
    result = asyncio.run(sandbox(handler).download_file("/tmp/empty", output))
    assert result.size_bytes == 0
    assert result.sha256 == hashlib.sha256(b"").hexdigest()
    assert output.getvalue() == b""


@pytest.mark.parametrize("status", [206, 302, 404, 409, 500])
def test_download_http_errors_never_copy_error_or_partial_response_bodies(status):
    stream = DownloadStream([b"not a complete file"])
    handler, seen = serving(httpx.Response(status, stream=stream))
    output = BytesIO()
    with pytest.raises(ServiceError) as raised:
        asyncio.run(sandbox(handler).download_file("/tmp/report", output))
    assert raised.value.code == {404: "file_not_found", 409: "file_changed"}.get(
        status, "computer_error"
    )
    assert output.getvalue() == b""
    assert stream.closed
    assert len(seen) == 1


@pytest.mark.parametrize("length", ["-1", "unknown", "1, 1", "9" * 5000])
def test_download_rejects_invalid_size_headers(length):
    stream = DownloadStream([b"data"])
    handler, _ = serving(
        httpx.Response(200, stream=stream, headers={"content-length": length})
    )
    output = BytesIO()
    with pytest.raises(ServiceError) as raised:
        asyncio.run(sandbox(handler).download_file("/tmp/report", output))
    assert raised.value.code == "invalid_file_download"
    assert output.getvalue() == b""
    assert stream.closed


@pytest.mark.parametrize("declared_length", [2, 8])
def test_download_rejects_mismatched_size_including_early_eof(declared_length):
    stream = DownloadStream([b"data"])
    handler, _ = serving(
        httpx.Response(
            200, stream=stream, headers={"content-length": str(declared_length)}
        )
    )
    output = BytesIO()
    with pytest.raises(ServiceError) as raised:
        asyncio.run(sandbox(handler).download_file("/tmp/report", output))
    assert raised.value.code == "invalid_file_download"
    assert stream.closed
    assert output.tell() == len(output.getvalue())


def test_download_rejects_oversized_header_before_reading():
    from kadan.api.sandbox_client import MAX_FILE_BYTES

    stream = DownloadStream(error=AssertionError("must not read oversized response"))
    handler, _ = serving(
        httpx.Response(
            200, stream=stream, headers={"content-length": str(MAX_FILE_BYTES + 1)}
        )
    )
    with pytest.raises(ServiceError) as raised:
        asyncio.run(sandbox(handler).download_file("/tmp/large", BytesIO()))
    assert raised.value.status == 413
    assert stream.closed


def test_download_enforces_stream_limit_without_a_size_header(monkeypatch):
    monkeypatch.setattr("kadan.api.sandbox_client.MAX_FILE_BYTES", 65536)
    stream = DownloadStream([b"a" * 65536, b"extra"])
    handler, _ = serving(httpx.Response(200, stream=stream))
    output = BytesIO()
    with pytest.raises(ServiceError) as raised:
        asyncio.run(sandbox(handler).download_file("/tmp/large", output))
    assert raised.value.status == 413
    assert output.getvalue() == b"a" * 65536
    assert output.tell() == 65536
    assert stream.closed


def test_download_rejects_unrequested_encoding_before_copying():
    stream = DownloadStream([b"encoded content"])
    handler, _ = serving(
        httpx.Response(200, stream=stream, headers={"content-encoding": "gzip"})
    )
    output = BytesIO()
    with pytest.raises(ServiceError) as raised:
        asyncio.run(sandbox(handler).download_file("/tmp/report", output))
    assert raised.value.code == "invalid_file_download"
    assert output.getvalue() == b""
    assert stream.closed


@pytest.mark.parametrize(
    "error",
    [httpx.ReadError("connection lost"), httpx.RemoteProtocolError("incomplete body")],
)
def test_download_stream_failure_closes_upstream_without_rewinding_partial_output(
    error,
):
    stream = DownloadStream([b"a" * 65536], error=error)
    handler, _ = serving(httpx.Response(200, stream=stream))
    output = BytesIO()
    with pytest.raises(ServiceError) as raised:
        asyncio.run(sandbox(handler).download_file("/tmp/report", output))
    assert raised.value.code == "computer_unavailable"
    assert output.tell() == 65536
    assert stream.closed
    assert not output.closed


def test_download_total_deadline_closes_stalled_stream(monkeypatch):
    monkeypatch.setattr("kadan.api.sandbox_client.OPERATION_SECONDS", 0.01)
    stream = DownloadStream(wait=True)
    handler, _ = serving(httpx.Response(200, stream=stream))
    with pytest.raises(ServiceError) as raised:
        asyncio.run(sandbox(handler).download_file("/tmp/report", BytesIO()))
    assert raised.value.code == "computer_unavailable"
    assert stream.closed


def test_download_cancellation_propagates_and_closes_upstream():
    stream = DownloadStream([b"a" * 65536], error=asyncio.CancelledError())
    handler, _ = serving(httpx.Response(200, stream=stream))
    output = BytesIO()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(sandbox(handler).download_file("/tmp/report", output))
    assert output.tell() == 65536
    assert stream.closed
    assert not output.closed


def upload_receipt(path="/tmp/destination", size=4):
    return {
        "success": True,
        "data": {"success": True, "file_path": path, "file_size": size},
    }


def test_file_upload_sends_seekable_binary_file_and_requires_full_receipt():
    handler, seen = serving(httpx.Response(200, json=upload_receipt()))
    source = BytesIO(b"\x00\xffab")
    source.seek(3)
    asyncio.run(sandbox(handler).upload_file("/tmp/destination", source))
    assert seen[0].method == "POST"
    assert seen[0].url.path == "/v1/file/upload"
    assert seen[0].headers["authorization"] == "Bearer secret"
    assert b'name="path"' in seen[0].content
    assert b"/tmp/destination" in seen[0].content
    assert b"\x00\xffab" in seen[0].content
    assert not source.closed


@pytest.mark.parametrize(
    "receipt",
    [
        {"success": False, "data": {"error_type": "permission_denied"}},
        {**upload_receipt(), "success": False},
        {"success": True, "data": {**upload_receipt()["data"], "success": False}},
        upload_receipt(path="/tmp/other"),
        upload_receipt(size=3),
        upload_receipt(size=True),
        {"success": True},
        {"data": upload_receipt()["data"]},
    ],
)
def test_file_upload_rejects_failed_or_mismatched_receipt(receipt):
    handler, seen = serving(httpx.Response(200, json=receipt))
    with pytest.raises(ServiceError) as raised:
        asyncio.run(sandbox(handler).upload_file("/tmp/destination", BytesIO(b"data")))
    assert raised.value.code == "invalid_file_upload"
    assert len(seen) == 1


def test_file_upload_rejects_non_json_receipt():
    handler, _ = serving(httpx.Response(200, text="not JSON"))
    with pytest.raises(ServiceError, match="invalid file upload receipt"):
        asyncio.run(sandbox(handler).upload_file("/tmp/destination", BytesIO(b"data")))


def test_file_upload_rejects_oversized_staging_before_sending(monkeypatch):
    monkeypatch.setattr("kadan.api.sandbox_client.MAX_FILE_BYTES", 3)
    handler, seen = serving()
    with pytest.raises(ServiceError) as raised:
        asyncio.run(sandbox(handler).upload_file("/tmp/destination", BytesIO(b"data")))
    assert raised.value.status == 413
    assert seen == []


def test_file_upload_deadline_never_retries(monkeypatch):
    monkeypatch.setattr("kadan.api.sandbox_client.OPERATION_SECONDS", 0.01)
    seen = []

    async def stalled(request):
        seen.append(request)
        await asyncio.Event().wait()

    with pytest.raises(ServiceError, match="did not confirm"):
        asyncio.run(sandbox(stalled).upload_file("/tmp/destination", BytesIO(b"data")))
    assert len(seen) == 1
