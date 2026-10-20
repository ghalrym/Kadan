import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import Request

from kadan.api.routes.computer import screenshot


def test_screenshot_is_relayed_without_caching():
    sandbox = SimpleNamespace(
        screenshot=AsyncMock(return_value=(b"jpeg", "image/jpeg"))
    )
    response = asyncio.run(
        screenshot.computer_screenshot(
            Request(
                {
                    "type": "http",
                    "app": SimpleNamespace(
                        state=SimpleNamespace(services=SimpleNamespace(sandbox=sandbox))
                    ),
                }
            )
        )
    )
    assert response.body == b"jpeg"
    assert response.media_type == "image/jpeg"
    assert response.headers["cache-control"] == "private, no-store"
