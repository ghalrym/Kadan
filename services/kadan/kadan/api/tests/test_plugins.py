import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, ErrorData, ListToolsResult, TextContent, Tool
from pydantic import ValidationError
from starlette.requests import Request

from kadan.api import plugins
from kadan.api.config import ServiceError
from kadan.api.plugins import PluginRegistry, PluginSession
from kadan.api.pydantic_models.plugins import (
    PluginManifest,
    StreamableHTTPServer,
    builtin_plugin,
)
from kadan.api.routes.plugins import plugins as plugin_catalog


def manifest(plugin_id="kadan", enabled=True):
    return builtin_plugin().model_copy(update={"id": plugin_id, "enabled": enabled})


def tool(name="read", schema=None):
    return Tool(
        name=name,
        inputSchema=schema
        if schema is not None
        else {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
            "additionalProperties": False,
        },
    )


def result(text="Saved", error=False):
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=error)


def fake_connection(monkeypatch, pages=None):
    session = SimpleNamespace(
        list_tools=AsyncMock(side_effect=pages or [ListToolsResult(tools=[tool()])]),
        call_tool=AsyncMock(return_value=result()),
        initialize=AsyncMock(),
    )
    state = SimpleNamespace(closed=False, server_calls=[])

    def server(services, key):
        state.server_calls.append((services, key))
        return object()

    @asynccontextmanager
    async def connect(server, **kwargs):
        try:
            yield session
        finally:
            state.closed = True

    monkeypatch.setattr(plugins, "create_builtin_server", server)
    monkeypatch.setattr(plugins, "create_connected_server_and_client_session", connect)
    return session, state


def test_selection_defaults_to_enabled_and_explicit_empty_means_no_tools():
    registry = PluginRegistry([manifest(), manifest("disabled", False)])
    assert registry.select(None) == ["kadan"]
    assert registry.select([]) == []
    for selected in [["missing"], ["disabled"], ["kadan", "kadan"]]:
        with pytest.raises(ServiceError) as raised:
            registry.select(selected)
        assert raised.value.code == "invalid_plugins"


def test_duplicate_ids_fail_and_catalog_cannot_mutate_live_configuration():
    with pytest.raises(ValueError, match="Duplicate plugin"):
        PluginRegistry([manifest(), manifest()])
    configured = manifest()
    registry = PluginRegistry([configured])
    configured.enabled = False
    catalog = registry.catalog()
    catalog[0].skills[0].instructions = "Changed"
    catalog[0].enabled = False
    assert registry.select(None) == ["kadan"]
    assert registry.catalog()[0].skills[0].instructions != "Changed"


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "https://u:p@example.com/mcp",
        "https://example.com/mcp?token=x",
        "http://example.com/mcp#secret",
        "http://example.com:0/mcp",
        "http://example.com:99999/mcp",
        "http://example.com/ white",
        "http:///mcp",
    ],
)
def test_endpoint_rejects_unsupported_transport_and_credentials(url):
    with pytest.raises(ValidationError):
        StreamableHTTPServer(url=url)


def test_manifest_rejects_subprocess_configuration_and_ambiguous_ids():
    document = builtin_plugin().model_dump()
    document["server"] = {"transport": "stdio", "command": "anything"}
    with pytest.raises(ValidationError):
        PluginManifest.model_validate(document)
    document = builtin_plugin().model_dump()
    document["id"] = "a__b"
    with pytest.raises(ValidationError):
        PluginManifest.model_validate(document)


def test_discovery_qualifies_paged_tools_and_binds_current_agent(monkeypatch):
    session, state = fake_connection(
        monkeypatch,
        [
            ListToolsResult(tools=[tool("first")], nextCursor="next"),
            ListToolsResult(tools=[tool("second")]),
        ],
    )
    services = object()

    async def run():
        async with PluginRegistry([manifest()]).connect(
            services, "agent-one", None
        ) as active:
            assert [item.name for item in active.tools] == [
                "kadan__first",
                "kadan__second",
            ]
            assert active.skills[0].instructions
            assert (await active.call_tool("kadan__second", {"id": 2})).content[
                0
            ].text == "Saved"
            assert not state.closed

    asyncio.run(run())
    assert state.server_calls == [(services, "agent-one")]
    assert state.closed
    assert session.list_tools.call_args_list[1].kwargs["params"].cursor == "next"
    assert session.call_tool.call_args.args == ("second", {"id": 2})


def test_empty_selection_never_opens_server(monkeypatch):
    _, state = fake_connection(monkeypatch)

    async def run():
        async with PluginRegistry([manifest()]).connect(
            object(), "agent", []
        ) as active:
            assert active.tools == active.skills == []

    asyncio.run(run())
    assert state.server_calls == []


@pytest.mark.parametrize(
    "pages",
    [
        [ListToolsResult(tools=[tool(), tool()])],
        [
            ListToolsResult(tools=[], nextCursor="same"),
            ListToolsResult(tools=[], nextCursor="same"),
        ],
        [
            ListToolsResult(
                tools=[
                    tool(
                        schema={"type": "object", "$ref": "https://example.com/schema"}
                    )
                ]
            )
        ],
        [ListToolsResult(tools=[tool(name="a" * 129)])],
    ],
)
def test_bad_discovery_fails_closed_and_closes_sessions(monkeypatch, pages):
    _, state = fake_connection(monkeypatch, pages)

    async def run():
        async with PluginRegistry([manifest()]).connect(object(), "agent", None):
            pytest.fail("Invalid tools must not be exposed")

    with pytest.raises(ServiceError) as raised:
        asyncio.run(run())
    assert raised.value.code == "invalid_plugin_tools"
    assert state.closed


def test_tool_validation_and_unknown_name_never_call_server():
    client = SimpleNamespace(call_tool=AsyncMock(return_value=result()))
    active = PluginSession()
    active.add_tool("test", client, tool())

    async def run():
        for name, arguments in [
            ("other__read", {"id": 1}),
            ("test__read", {}),
            ("test__read", {"id": "bad"}),
            ("test__read", {"id": 1, "agent_key": "other"}),
            ("test__read", {"id": float("nan")}),
            ("test__read", []),
        ]:
            assert (await active.call_tool(name, arguments)).isError

    asyncio.run(run())
    client.call_tool.assert_not_awaited()


def test_local_schema_references_validate_without_network():
    active = PluginSession()
    client = SimpleNamespace(call_tool=AsyncMock(return_value=result()))
    active.add_tool(
        "test",
        client,
        tool(
            schema={
                "type": "object",
                "$defs": {"id": {"type": "integer"}},
                "properties": {"id": {"$ref": "#/$defs/id"}},
                "required": ["id"],
            }
        ),
    )
    assert not asyncio.run(active.call_tool("test__read", {"id": 1})).isError
    assert asyncio.run(active.call_tool("test__read", {"id": "bad"})).isError
    assert client.call_tool.await_count == 1


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("private endpoint details"),
        McpError(ErrorData(code=-32602, message="private endpoint details")),
    ],
)
def test_call_failure_returns_safe_error_with_write_uncertainty(failure):
    active = PluginSession()
    active.add_tool(
        "test", SimpleNamespace(call_tool=AsyncMock(side_effect=failure)), tool()
    )
    output = asyncio.run(active.call_tool("test__read", {"id": 1}))
    assert output.isError
    assert "write may already have completed" in output.content[0].text
    assert "private endpoint" not in output.content[0].text


def test_oversized_result_does_not_silently_truncate_or_claim_write_failed():
    active = PluginSession()
    active.add_tool(
        "test",
        SimpleNamespace(call_tool=AsyncMock(return_value=result("x" * 32001))),
        tool(),
    )
    output = asyncio.run(active.call_tool("test__read", {"id": 1}))
    assert output.isError
    assert "write may already have completed" in output.content[0].text


def test_cancellation_propagates_and_closes_session(monkeypatch):
    client, state = fake_connection(monkeypatch)
    client.call_tool.side_effect = asyncio.CancelledError()

    async def run():
        async with PluginRegistry([manifest()]).connect(
            object(), "agent", None
        ) as active:
            await active.call_tool("kadan__read", {"id": 1})

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
    assert state.closed


def test_caller_error_retains_type_when_sdk_wraps_body_errors(monkeypatch):
    _, state = fake_connection(monkeypatch)

    @asynccontextmanager
    async def sdk_group(*args, **kwargs):
        try:
            yield SimpleNamespace(
                list_tools=AsyncMock(return_value=ListToolsResult(tools=[]))
            )
        except ServiceError as error:
            raise ExceptionGroup("SDK task group", [error])
        finally:
            state.closed = True

    monkeypatch.setattr(
        plugins, "create_connected_server_and_client_session", sdk_group
    )
    expected = ServiceError("Conversation changed", 409, "version_conflict")

    async def run():
        async with PluginRegistry([manifest()]).connect(object(), "agent", None):
            raise expected

    with pytest.raises(ServiceError) as raised:
        asyncio.run(run())
    assert raised.value is expected
    assert state.closed


def test_catalog_route_returns_metadata_without_discovery():
    registry = PluginRegistry([manifest()])
    request = Request(
        {
            "type": "http",
            "app": SimpleNamespace(
                state=SimpleNamespace(services=SimpleNamespace(plugins=registry))
            ),
        }
    )
    catalog = asyncio.run(plugin_catalog(request))
    assert catalog[0].id == "kadan"
    assert catalog[0].setup.instructions


def test_http_transport_uses_only_configured_endpoint_and_no_environment_credentials(
    monkeypatch,
):
    state = {}
    client = SimpleNamespace(
        initialize=AsyncMock(),
        list_tools=AsyncMock(return_value=ListToolsResult(tools=[tool()])),
    )

    @asynccontextmanager
    async def http_client(**kwargs):
        state["http_options"] = kwargs
        yield "http-client"

    @asynccontextmanager
    async def transport(url, **kwargs):
        state["url"] = url
        state["transport_options"] = kwargs
        yield "read", "write", None

    @asynccontextmanager
    async def session(*args, **kwargs):
        assert args == ("read", "write")
        yield client

    monkeypatch.setattr(plugins.httpx, "AsyncClient", http_client)
    monkeypatch.setattr(plugins, "streamable_http_client", transport)
    monkeypatch.setattr(plugins, "ClientSession", session)
    configured = manifest("external").model_copy(
        update={"server": StreamableHTTPServer(url="https://example.test/mcp")}
    )

    async def run():
        async with PluginRegistry([configured]).connect(
            object(), "agent", None
        ) as active:
            assert active.tools[0].name == "external__read"

    asyncio.run(run())
    assert state["url"] == "https://example.test/mcp"
    assert state["http_options"]["trust_env"] is False
    assert state["http_options"]["follow_redirects"] is False
    assert state["transport_options"] == {"http_client": "http-client"}
    client.initialize.assert_awaited_once()


@pytest.mark.parametrize("operation", ["success", "caller_error", "cancel"])
def test_in_memory_sdk_adapter_preserves_results_errors_and_cancellation(
    monkeypatch, operation
):
    server = FastMCP("Unit fixture")

    @server.tool()
    async def read(id: int) -> str:
        return f"Read {id}"

    monkeypatch.setattr(plugins, "create_builtin_server", lambda services, key: server)
    expected = ServiceError("Conversation changed", 409, "version_conflict")

    async def run():
        async with PluginRegistry([manifest()]).connect(
            object(), "agent", None
        ) as active:
            if operation == "caller_error":
                raise expected
            if operation == "cancel":
                asyncio.current_task().cancel()
                await asyncio.sleep(0)
            output = await active.call_tool("kadan__read", {"id": 3})
            assert output.content[0].text == "Read 3"
            assert not output.isError

    if operation == "cancel":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(run())
    elif operation == "caller_error":
        with pytest.raises(ServiceError) as raised:
            asyncio.run(run())
        assert raised.value is expected
    else:
        asyncio.run(run())


def test_compliant_mcp_tool_names_keep_dots_and_full_length():
    active = PluginSession()
    client = SimpleNamespace(call_tool=AsyncMock(return_value=result()))
    active.add_tool("test", client, tool("documents.read"))
    active.add_tool("test", client, tool("a" * 128))
    assert [item.name for item in active.tools] == [
        "test__documents.read",
        "test__" + "a" * 128,
    ]


def test_in_memory_sdk_cancels_running_tool_and_closes_its_handler(monkeypatch):
    server = FastMCP("Cancellation fixture")
    started = asyncio.Event()
    closed = asyncio.Event()

    @server.tool()
    async def read(id: int) -> str:
        try:
            started.set()
            await asyncio.Event().wait()
            return str(id)
        finally:
            closed.set()

    monkeypatch.setattr(plugins, "create_builtin_server", lambda services, key: server)

    async def run():
        async def operation():
            async with PluginRegistry([manifest()]).connect(
                object(), "agent", None
            ) as active:
                await active.call_tool("kadan__read", {"id": 1})

        task = asyncio.create_task(operation())
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()

    asyncio.run(run())


def test_discovery_timeout_is_actionable_and_cleans_up(monkeypatch):
    session, state = fake_connection(monkeypatch)

    async def blocked(**kwargs):
        await asyncio.Event().wait()

    session.list_tools.side_effect = blocked
    monkeypatch.setattr(plugins, "DISCOVERY_TIMEOUT_SECONDS", 0.01)

    async def run():
        async with PluginRegistry([manifest()]).connect(object(), "agent", None):
            pytest.fail("Timed out discovery must not expose tools")

    with pytest.raises(ServiceError) as raised:
        asyncio.run(run())
    assert raised.value.code == "plugin_unavailable"
    assert state.closed


def test_call_timeout_ends_turn_and_cancels_pending_request(monkeypatch):
    closed = []

    async def blocked(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            closed.append(True)

    active = PluginSession()
    active.add_tool("test", SimpleNamespace(call_tool=blocked), tool())
    monkeypatch.setattr(plugins, "TOOL_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(ServiceError) as raised:
        asyncio.run(active.call_tool("test__read", {"id": 1}))
    assert raised.value.code == "tool_timeout"
    assert "write is unknown" in str(raised.value)
    assert closed == [True]


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError(),
        httpx.ReadTimeout("private endpoint details"),
        McpError(ErrorData(code=408, message="private endpoint details")),
    ],
)
def test_transport_and_sdk_timeouts_are_terminal(failure):
    active = PluginSession()
    active.add_tool(
        "test", SimpleNamespace(call_tool=AsyncMock(side_effect=failure)), tool()
    )
    with pytest.raises(ServiceError) as raised:
        asyncio.run(active.call_tool("test__read", {"id": 1}))
    assert raised.value.code == "tool_timeout"
    assert raised.value.status == 504
    assert "write is unknown" in str(raised.value)
    assert "private endpoint" not in str(raised.value)


def test_in_memory_sdk_timeout_closes_running_handler_before_propagating(monkeypatch):
    server = FastMCP("Timeout fixture")
    events = []

    @server.tool()
    async def read(id: int) -> str:
        try:
            events.append("started")
            await asyncio.Event().wait()
            return str(id)
        finally:
            events.append("closed")

    monkeypatch.setattr(plugins, "create_builtin_server", lambda services, key: server)
    monkeypatch.setattr(plugins, "TOOL_TIMEOUT_SECONDS", 0.05)

    async def run():
        with pytest.raises(ServiceError) as raised:
            async with PluginRegistry([manifest()]).connect(
                object(), "agent", None
            ) as active:
                await active.call_tool("kadan__read", {"id": 1})
                events.append("next planner or call")
        assert raised.value.code == "tool_timeout"
        assert "write is unknown" in str(raised.value)
        assert events == ["started", "closed"]

    asyncio.run(run())


def test_argument_validation_error_is_bounded():
    active = PluginSession()
    active.add_tool("test", SimpleNamespace(call_tool=AsyncMock()), tool())
    output = asyncio.run(active.call_tool("test__read", {"id": "x" * 90000}))
    assert output.isError
    assert len(output.model_dump_json()) < 2000
