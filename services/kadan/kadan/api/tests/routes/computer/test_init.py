import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import Request

from kadan.api.config import ServiceError
from kadan.api.routes import computer


def request(sandbox):
    return Request(
        {
            "type": "http",
            "app": SimpleNamespace(
                state=SimpleNamespace(services=SimpleNamespace(sandbox=sandbox))
            ),
        }
    )


@pytest.mark.parametrize(
    ("sandbox", "state", "message"),
    [
        (SimpleNamespace(configured=False), "not_configured", "No computer"),
        (
            SimpleNamespace(
                configured=True,
                context=AsyncMock(side_effect=ServiceError("Computer is down")),
            ),
            "unavailable",
            "Computer is down",
        ),
        (
            SimpleNamespace(configured=True, context=AsyncMock(return_value={})),
            "ready",
            "Ready",
        ),
    ],
)
def test_status_reports_whether_the_computer_answers(sandbox, state, message):
    status = asyncio.run(computer.computer_status(request(sandbox)))
    assert status.state == state
    assert status.message.startswith(message)
