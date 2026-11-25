"""Screenshot tool and availability tests with in-memory HTTP/computer boundaries."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from kadan.api.config import ServiceError, Settings
from kadan.runner.agent_runs import AgentRuns
from kadan.runner.tools import ALL_TOOLS, COMPUTER_AGENT_TOOLS, VISION_TOOLS
from kadan.runner.tools.vision import computer_look


def chat_response(model="inkling-2-mini-iq4-nl", content="A Save button is visible."):
    return httpx.Response(
        200,
        json={
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        },
    )


def computer(response=None):
    settings = Settings(_env_file=None)
    client = SimpleNamespace(post=AsyncMock(return_value=response or chat_response()))
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
    assert result == {
        "model": "inkling-2-mini-iq4-nl",
        "body": "A Save button is visible.",
    }
    call = target.services.client.post.call_args
    assert call.args == (target.services.settings.hikari_url + "/v1/chat/completions",)
    body = call.kwargs["json"]
    assert body["stream"] is False
    assert "model" not in body
    assert body["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Where is Save?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,c2NyZWVuc2hvdA=="},
                },
            ],
        }
    ]


@pytest.mark.parametrize(
    "model",
    [
        "inkling-2-mini-iq4-nl",
        "inkling-2-mini-q8-0",
        "claude-haiku-5-5",
        "gpt-6.1-terra",
    ],
)
def test_local_and_subscription_observations_are_accepted(model):
    target = computer(response=chat_response(model=model))
    assert invoke(target) == {"model": model, "body": "A Save button is visible."}


def test_unknown_model_is_rejected():
    target = computer(response=chat_response(model="unknown"))
    with pytest.raises(ServiceError, match="invalid screenshot description"):
        invoke(target)


def test_worker_failure_is_reported_without_fabricating_an_observation():
    with pytest.raises(ServiceError, match="HTTP 503"):
        invoke(computer(response=httpx.Response(503)))


def test_vision_belongs_to_the_computer_agent():
    owner = SimpleNamespace(
        services=SimpleNamespace(
            settings=Settings(_env_file=None),
            sandbox=SimpleNamespace(configured=True),
        )
    )
    run = SimpleNamespace(references=[])
    assert AgentRuns.bound_tools(owner, run) == [
        tool for tool in ALL_TOOLS if tool.name != "read_request_references"
    ]
    assert not any(tool in VISION_TOOLS for tool in AgentRuns.bound_tools(owner, run))
    assert all(tool in COMPUTER_AGENT_TOOLS for tool in VISION_TOOLS)
