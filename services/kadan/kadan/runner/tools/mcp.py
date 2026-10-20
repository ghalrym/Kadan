"""MCP tools: the MCP servers that run inside the computer."""

from typing import Any

from pydantic import Field

from ...api.sandbox_client import SandboxClient, segment
from .base import Arguments, body, computer_tool


class ListServersArguments(Arguments):
    include_hidden: bool = Field(default=False, description="Include hidden servers.")


@computer_tool("GET")
async def mcp_list_mcp_servers(
    args: ListServersArguments, computer: SandboxClient
) -> dict:
    """List the configured MCP servers."""
    return await computer.run("GET", "/v1/mcp/servers", params=body(args))


class ServerArguments(Arguments):
    server_name: str = Field(description="MCP server name from mcp_list_mcp_servers.")


@computer_tool("GET")
async def mcp_list_mcp_tools(args: ServerArguments, computer: SandboxClient) -> dict:
    """List an MCP server's tools with their descriptions and parameters."""
    return await computer.run("GET", f"/v1/mcp/{segment(args.server_name)}/tools")


class ExecuteToolArguments(ServerArguments):
    tool_name: str = Field(description="Tool name from mcp_list_mcp_tools.")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="The tool's arguments."
    )


@computer_tool("POST")
async def mcp_execute_mcp_tool(
    args: ExecuteToolArguments, computer: SandboxClient
) -> dict:
    """Run a tool on one of the computer's MCP servers."""
    return await computer.run(
        "POST",
        f"/v1/mcp/{segment(args.server_name)}/tools/{segment(args.tool_name)}",
        json_body=args.arguments,
    )


TOOLS = [
    # mcp_list_mcp_servers,
    # mcp_list_mcp_tools,
    # mcp_execute_mcp_tool,
]
