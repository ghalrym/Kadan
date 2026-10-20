import asyncio

import pytest

from kadan.api.http_safety import LocalRequestSafety, body_limit, same_origin


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
        (b"192.0.2.11:8630", b"http://192.0.2.11:8630", 204),
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


def test_only_pet_imports_accept_large_bodies():
    assert body_limit("/api/agents/a1/pets/imports") == 64 * 1024 * 1024
    for path in ("/api/agents/a1/pets", "/api/agents/a1/pets/imports/x", "/api/x"):
        assert body_limit(path) == 1024 * 1024


@pytest.mark.parametrize(
    ("headers", "accepted"),
    [
        ([(b"host", b"127.0.0.1:8630"), (b"origin", b"http://127.0.0.1:8630")], True),
        ([(b"host", b"127.0.0.1:8630")], False),
        ([(b"host", b"127.0.0.1:8630"), (b"origin", b"http://evil.example")], False),
        ([(b"host", b"127.0.0.1:8630"), (b"origin", b"null")], False),
        ([(b"host", b"evil.example"), (b"origin", b"http://evil.example")], False),
        ([(b"host", b"192.0.2.10:8630"), (b"origin", b"http://192.0.2.10:8630")], True),
        ([(b"host", b"192.0.2.10:8630"), (b"origin", b"http://evil.example")], False),
        (
            [(b"host", b"192.0.2.10:8630"), (b"origin", b"http://192.0.2.10:8620")],
            False,
        ),
        ([(b"host", b"192.0.2.10:8630")], False),
        (
            [
                (b"host", b"[2001:db8::1]:8630"),
                (b"origin", b"http://[2001:db8::1]:8630"),
            ],
            True,
        ),
    ],
)
def test_websockets_require_an_allowed_host_and_this_exact_origin(headers, accepted):
    called = []
    sent = []

    async def app(scope, receive, send):
        called.append(scope)

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        sent.append(message)

    asyncio.run(
        LocalRequestSafety(app, ["127.0.0.1", "localhost", "::1"])(
            {"type": "websocket", "scheme": "ws", "headers": headers},
            receive,
            send,
        )
    )
    assert bool(called) is accepted
    assert sent == ([] if accepted else [{"type": "websocket.close", "code": 1008}])


@pytest.mark.parametrize(
    ("host", "origin", "expected"),
    [
        ("192.0.2.1:8630", "http://192.0.2.1:8630", 204),
        ("[2001:db8::1]:8630", "http://[2001:db8::1]:8630", 204),
        ("kadan.home:8630", "http://kadan.home:8630", 204),
        ("evil.example:8630", "http://evil.example:8630", 400),
        ("192.0.2.1.evil.example:8630", "http://192.0.2.1.evil.example:8630", 400),
        ("192.0.2.1:8630", "http://evil.example", 403),
        ("192.0.2.1:8630", "http://192.0.2.1:8620", 403),
        ("192.0.2.1:8630", "https://192.0.2.1:8630", 403),
        ("@192.0.2.1:8630", None, 400),
        ("user:password@192.0.2.1:8630", None, 400),
        ("192.0.2.1:0", None, 400),
        ("192.0.2.1:65536", None, 400),
        ("192.0.2.1:bad", None, 400),
        ("192.0.2.1/path", None, 400),
        ("192.0.2.1?query=x", None, 400),
        ("192.0.2.1#fragment", None, 400),
        ("[2001:db8::1", None, 400),
    ],
)
def test_ip_hosts_and_explicit_dns_keep_origin_and_host_validation(
    host, origin, expected
):
    sent = []

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        sent.append(message)

    headers = [(b"host", host.encode())]
    if origin is not None:
        headers.append((b"origin", origin.encode()))
    asyncio.run(
        LocalRequestSafety(app, ["kadan.home"])(
            {"type": "http", "method": "POST", "scheme": "http", "headers": headers},
            receive,
            send,
        )
    )
    assert sent[0]["status"] == expected
