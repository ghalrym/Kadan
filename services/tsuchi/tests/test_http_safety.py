import pytest

from tsuchi.http_safety import allowed_request_origin


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
