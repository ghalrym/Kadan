import asyncio

import pytest
from tsuchi.http_safety import LocalRequestSafety, allowed_request_origin


@pytest.mark.parametrize(
    ("origin", "host", "scheme", "expected"),
    [
        ("http://localhost:8710", "localhost:8710", "http", True),
        ("http://localhost", "localhost:80", "http", True),
        ("http://[::1]:8710", "[::1]:8710", "http", True),
        ("http://localhost:8711", "localhost:8710", "http", False),
        ("https://localhost:8710", "localhost:8710", "http", False),
        ("https://elsewhere.example", "localhost:8710", "http", False),
        ("http://localhost:8710/path", "localhost:8710", "http", False),
        ("null", "localhost:8710", "http", False),
        ("http://user@localhost:8710", "localhost:8710", "http", False),
    ],
)
def test_write_origin_must_match_scheme_host_and_port(origin, host, scheme, expected):
    assert allowed_request_origin(origin, host, scheme) is expected


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
        LocalRequestSafety(app, ["kadan.home"], max_body_bytes=1024)(
            {"type": "http", "method": "POST", "scheme": "http", "headers": headers},
            receive,
            send,
        )
    )
    assert sent[0]["status"] == expected
