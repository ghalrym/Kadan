import asyncio

import pytest

from kadan.api.http_safety import LocalRequestSafety, same_origin


@pytest.mark.parametrize(
    "origin",
    [
        "http://evil.example",
        "http://127.0.0.1:9999",
        "null",
        "http://user@127.0.0.1:8630",
        "http://127.0.0.1:8630/path",
    ],
)
def test_cross_origin_writes_are_not_same_origin(origin):
    assert not same_origin(origin, "127.0.0.1:8630", "http")


def test_exact_loopback_origin_is_allowed():
    assert same_origin("http://127.0.0.1:8630", "127.0.0.1:8630", "http")


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ([(b"host", b"evil.example")], 400),
        ([(b"host", b"@127.0.0.1:8630")], 400),
        ([(b"host", b"127.0.0.1:0")], 400),
        ([(b"host", b"127.0.0.1:8630"), (b"origin", b"http://evil.example")], 403),
        ([(b"host", b"127.0.0.1:8630"), (b"sec-fetch-site", b"cross-site")], 403),
        ([(b"host", b"127.0.0.1:8630"), (b"content-length", b"1048577")], 413),
    ],
)
def test_local_guard_rejects_request_before_app_is_called(headers, expected):
    sent = []

    async def app(scope, receive, send):
        raise AssertionError("Guard should stop this request")

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        sent.append(message)

    asyncio.run(
        LocalRequestSafety(app, ["127.0.0.1", "localhost", "::1"])(
            {"type": "http", "method": "POST", "scheme": "http", "headers": headers},
            receive,
            send,
        )
    )
    assert sent[0]["status"] == expected


@pytest.mark.parametrize(
    ("host", "origin", "expected"),
    [
        (b"192.0.2.10:8630", b"http://192.0.2.10:8630", 204),
        (b"192.0.2.11:8630", b"http://192.0.2.11:8630", 400),
        (b"192.0.2.10:8630", b"http://192.0.2.11:8630", 403),
        (b"192.0.2.10:8630", b"http://192.0.2.10:8620", 403),
    ],
)
def test_lan_writes_require_an_allowed_host_and_same_origin(host, origin, expected):
    sent = []

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        sent.append(message)

    asyncio.run(
        LocalRequestSafety(app, ["192.0.2.10"])(
            {
                "type": "http",
                "method": "POST",
                "scheme": "http",
                "headers": [(b"host", host), (b"origin", origin)],
            },
            receive,
            send,
        )
    )
    assert sent[0]["status"] == expected
