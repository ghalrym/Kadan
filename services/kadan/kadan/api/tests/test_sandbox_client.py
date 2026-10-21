"""Sandbox client tests against an in-memory HTTP transport, never a real computer."""

import asyncio
import json
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
