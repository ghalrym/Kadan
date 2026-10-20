import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request

from kadan.api.routes.status import (
    InferenceState,
    ServiceState,
    Status,
    browser_link,
    status,
)


@pytest.mark.parametrize(
    ("url", "hostname", "expected"),
    [
        ("http://127.0.0.1:8620", "192.0.2.1", "http://192.0.2.1:8620"),
        (
            "http://localhost:18088/settings",
            "kadan.home",
            "http://kadan.home:18088/settings",
        ),
        (
            "http://[::1]:8088/settings",
            "2001:db8::1",
            "http://[2001:db8::1]:8088/settings",
        ),
        ("http://127.0.0.1:8620", "localhost", "http://localhost:8620"),
        ("http://storage.home:8620", "kadan.home", "http://storage.home:8620"),
        (
            "http://192.0.2.2:8088/settings",
            "192.0.2.1",
            "http://192.0.2.2:8088/settings",
        ),
    ],
)
def test_operator_links_follow_browser_host_but_keep_explicit_remote_urls(
    url, hostname, expected
):
    assert browser_link(url, hostname) == expected


def test_status_projects_links_without_mutating_shared_service_state_or_using_forwarded_host():
    original = Status(
        tsuchi=ServiceState(
            state="ready", message="Ready", browser_url="http://127.0.0.1:8620"
        ),
        hikari=InferenceState(
            state="model_required",
            message="Choose a model",
            settings_url="http://127.0.0.1:8088/settings",
        ),
    )
    services = SimpleNamespace(status=AsyncMock(return_value=original))
    request = Request(
        {
            "type": "http",
            "scheme": "http",
            "path": "/api/status",
            "query_string": b"",
            "headers": [
                (b"host", b"192.0.2.1:8630"),
                (b"x-forwarded-host", b"evil.example"),
            ],
            "app": SimpleNamespace(state=SimpleNamespace(services=services)),
        }
    )
    result = asyncio.run(status(request))
    assert result.tsuchi.browser_url == "http://192.0.2.1:8620"
    assert result.hikari.settings_url == "http://192.0.2.1:8088/settings"
    assert original.tsuchi.browser_url == "http://127.0.0.1:8620"
    assert original.hikari.settings_url == "http://127.0.0.1:8088/settings"
