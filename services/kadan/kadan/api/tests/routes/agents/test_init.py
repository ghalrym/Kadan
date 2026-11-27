import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, sentinel

import pytest
from fastapi import Request
from pydantic import ValidationError

from kadan.api.routes.agents import (
    AgentInput,
    AgentRename,
    delete_agent,
    get_agent,
    list_agents,
    rename_agent,
    router,
)


def test_agent_key_cannot_escape_its_namespace():
    with pytest.raises(ValidationError):
        AgentInput(key="../../another-agent", name="Name")


@pytest.mark.parametrize(
    "value",
    [
        {"name": " ", "expected_name": "Old"},
        {"name": "x" * 501, "expected_name": "Old"},
        {"name": "New"},
        {"name": "New", "expected_name": "Old", "key": "changed"},
    ],
)
def test_agent_rename_requires_valid_name_and_previous_name(value):
    with pytest.raises(ValidationError):
        AgentRename(**value)


def test_agent_name_routes_proxy_exact_expected_name_and_saved_agent():
    json_request = AsyncMock(return_value=sentinel.saved_agent)
    request = Request(
        {
            "type": "http",
            "app": SimpleNamespace(
                state=SimpleNamespace(
                    services=SimpleNamespace(json_request=json_request)
                )
            ),
        }
    )
    assert asyncio.run(get_agent("ember", request)) is sentinel.saved_agent
    json_request.assert_awaited_once_with("Tsuchi", "GET", "/api/agents/ember")
    json_request.reset_mock()
    assert (
        asyncio.run(
            rename_agent(
                "ember", AgentRename(name=" New ", expected_name=" Old "), request
            )
        )
        is sentinel.saved_agent
    )
    json_request.assert_awaited_once_with(
        "Tsuchi",
        "PUT",
        "/api/agents/ember",
        json={"name": "New", "expected_name": " Old "},
    )


def test_list_agents_route_delegates_to_services_bootstrap_helper():
    list_helper = AsyncMock(return_value=[{"key": "kadan", "name": "Kadan Agent"}])
    request = Request(
        {
            "type": "http",
            "app": SimpleNamespace(
                state=SimpleNamespace(
                    services=SimpleNamespace(list_agents=list_helper)
                )
            ),
        }
    )
    assert asyncio.run(list_agents(request)) == [
        {"key": "kadan", "name": "Kadan Agent"}
    ]
    list_helper.assert_awaited_once_with()


def test_delete_agent_route_proxies_delete_to_tsuchi_with_204():
    json_request = AsyncMock(return_value=None)
    request = Request(
        {
            "type": "http",
            "app": SimpleNamespace(
                state=SimpleNamespace(
                    services=SimpleNamespace(json_request=json_request)
                )
            ),
        }
    )
    route = next(
        r
        for r in router.routes
        if r.path == "/api/agents/{key}" and "DELETE" in r.methods
    )
    assert route.status_code == 204
    assert asyncio.run(delete_agent("kadan", request)) is None
    json_request.assert_awaited_once_with("Tsuchi", "DELETE", "/api/agents/kadan")
