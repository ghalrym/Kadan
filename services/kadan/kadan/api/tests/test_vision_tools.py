"""Screenshot tool and availability tests with in-memory HTTP/computer boundaries."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from kadan.api.config import ServiceError, Settings
from kadan.runner.agent_runs import AgentRuns
from kadan.runner.tools import ALL_TOOLS, VISION_TOOLS
from kadan.runner.tools.vision import computer_look


def computer(response=None):
    settings = Settings(_env_file=None)
    client = SimpleNamespace(
        post=AsyncMock(
            return_value=response
            or httpx.Response(
                200,
                json={"model": "lfm3-vl-2b-bf16", "text": "A Save button is visible."},
            )
        )
    )
    return SimpleNamespace(
        services=SimpleNamespace(settings=settings, client=client),
        screenshot=AsyncMock(return_value=(b"screenshot", "image/jpeg")),
    )


def invoke(target):
    return asyncio.run(
        computer_look.ainvoke(
            {"question": "Where is Save?"},
            config={"configurable": {"computer": target}},
        )
    )


def test_tool_uses_fresh_screenshot_and_only_returns_a_bounded_observation():
    target = computer()
    result = invoke(target)
    target.screenshot.assert_awaited_once()
    assert result == {"model": "lfm3-vl-2b-bf16", "body": "A Save button is visible."}
    call = target.services.client.post.call_args
    assert call.args == (target.services.settings.hikari_url + "/v1/vision/describe",)
    assert call.kwargs["json"]["image_base64"] == "c2NyZWVuc2hvdA=="


def test_worker_failure_is_reported_without_fabricating_an_observation():
    with pytest.raises(ServiceError, match="HTTP 503"):
        invoke(computer(response=httpx.Response(503)))


def test_vision_is_available_only_when_the_agent_has_a_computer():
    owner = SimpleNamespace(
        services=SimpleNamespace(
            settings=Settings(_env_file=None),
            sandbox=SimpleNamespace(configured=True),
        )
    )
    assert AgentRuns.bound_tools(owner) == [*ALL_TOOLS, *VISION_TOOLS]
    owner.services.sandbox.configured = False
    assert not any(tool in VISION_TOOLS for tool in AgentRuns.bound_tools(owner))
