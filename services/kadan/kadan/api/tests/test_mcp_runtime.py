"""Unit tests for runner dynamic MCP tool binding, replay safety, and T3 watch registration."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from uuid import uuid4

import httpx

from kadan.api.config import Settings
from kadan.api.pydantic_models.agent_runs import ToolCommand
from kadan.api.pydantic_models.mcp import McpServerRecord, McpToolInfo
from kadan.api.services import LocalServices
from kadan.runner.agent_runs import validate
from kadan.runner.mcp_runtime import (
    MAX_MCP_RESULT_CHARS,
    McpOrigin,
    build_mcp_tools,
    execute_mcp_tool,
    search_mcp_tools,
)


def make_t3_server(pinned: list[str] | None = None) -> McpServerRecord:
    now = datetime.now(UTC)
    return McpServerRecord(
        id=uuid4(),
        slug="t3-code",
        name="T3 Code",
        preset="t3_code",
        url="https://t3.example/mcp",
        app_url="https://t3.example",
        auth_type="oauth",
        enabled=True,
        status="connected",
        pinned_tools=pinned if pinned is not None else ["t3_project_list", "t3_thread_launch"],
        tools=[
            McpToolInfo(
                name="t3_project_list",
                description="List projects",
                input_schema={"type": "object", "properties": {}},
                read_only=True,
            ),
            McpToolInfo(
                name="t3_thread_launch",
                description="Launch a thread",
                input_schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "message": {"type": "string"},
                        "clientRequestId": {"type": "string"},
                    },
                    "required": ["title"],
                },
                read_only=False,
                accepts_client_request_id=True,
            ),
            McpToolInfo(
                name="pull_request_list",
                description="List pull requests in a repository",
                input_schema={"type": "object", "properties": {"state": {"type": "string"}}},
                read_only=True,
            ),
        ],
        tools_synced_at=now,
        version=1,
        created_at=now,
        updated_at=now,
    )


def test_build_mcp_tools_binds_pinned_plus_gateway_and_validates_schema():
    server = make_t3_server()
    tools, registry = build_mcp_tools([server], reserved_names={"send_message"})
    names = {tool.name for tool in tools}
    assert names == {
        "t3_project_list",
        "t3_thread_launch",
        "mcp_search_tools",
        "mcp_call_tool",
    }
    assert set(registry) == {"t3_project_list", "t3_thread_launch"}

    # Missing required field produces ArgumentRejection instead of crashing
    rejection = validate(ToolCommand(tool="t3_thread_launch", arguments={}), tools)
    assert rejection is not None
    assert rejection.problems[0].field == "title"
    assert rejection.problems[0].code == "missing"

    # Valid call passes validation
    assert (
        validate(ToolCommand(tool="t3_thread_launch", arguments={"title": "Fix"}), tools)
        is None
    )


def test_search_mcp_tools_finds_unpinned_tools():
    server = make_t3_server()
    found = search_mcp_tools([server], query="pull request")
    assert len(found["matches"]) == 1
    assert found["matches"][0]["tool"] == "pull_request_list"
    assert found["matches"][0]["server"] == "t3-code"


def test_execute_mcp_tool_injects_client_request_id_registers_watch_and_bounds_replay():
    server = make_t3_server()
    watches_posted: list[dict] = []
    rpc_calls: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "tsuchi.test" and request.url.path == "/api/t3-watches":
            body = json.loads(request.content)
            watches_posted.append(body)
            return httpx.Response(201, json={**body, "status": "watching", "version": 1, "created_at": datetime.now(UTC).isoformat(), "updated_at": datetime.now(UTC).isoformat()})
        if request.url.path == "/mcp":
            rpc = json.loads(request.content)
            method = rpc.get("method")
            if method == "initialize":
                return httpx.Response(200, json={"jsonrpc": "2.0", "id": rpc["id"], "result": {}})
            if method == "notifications/initialized":
                return httpx.Response(202)
            if method == "tools/call":
                name = rpc["params"]["name"]
                args = rpc["params"]["arguments"]
                rpc_calls.append((name, args))
                if name == "t3_thread_launch":
                    payload = {
                        "threadId": "th-42",
                        "runId": "run-7",
                        "link": "https://t3.example/threads/th-42",
                        "title": args["title"],
                    }
                else:
                    payload = {"huge": "x" * (MAX_MCP_RESULT_CHARS + 5000)}
                return httpx.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": rpc["id"],
                        "result": {"content": [{"type": "text", "text": json.dumps(payload)}]},
                    },
                )
        return httpx.Response(404)

    async def scenario():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        services = LocalServices(
            client, Settings(tsuchi_url="http://tsuchi.test", hikari_url="http://hikari.test")
        )
        action_id = uuid4()
        origin = McpOrigin(agent_key="mi", conversation_id=uuid4(), run_id=uuid4())
        launch_info = server.tools[1]

        # Even on replayed=True, clientRequestId injection makes t3_thread_launch replay-safe!
        res = await execute_mcp_tool(
            services,
            server,
            launch_info,
            {"title": "Add feature", "message": "Implement it"},
            action_id=action_id,
            replayed=True,
            origin=origin,
        )
        assert res["background_watch"] is True
        assert rpc_calls[0][1]["clientRequestId"] == str(action_id)
        assert len(watches_posted) == 1
        assert watches_posted[0]["thread_id"] == "th-42"
        assert watches_posted[0]["run_id"] == "run-7"

        # Non-idempotent tool without clientRequestId is blocked on replay
        non_idem = McpToolInfo(name="send_email", description="Send email", read_only=False)
        blocked = await execute_mcp_tool(
            services, server, non_idem, {}, action_id=uuid4(), replayed=True, origin=origin
        )
        assert "restarted" in blocked["error"]

        # Oversized MCP results are cleanly truncated
        big = await execute_mcp_tool(
            services, server, server.tools[0], {}, action_id=uuid4(), replayed=False
        )
        assert big["truncated"] is True
        assert len(big["excerpt"]) <= MAX_MCP_RESULT_CHARS
        await client.aclose()

    asyncio.run(scenario())
