"""MCP gateway tools for discovering and calling unpinned remote MCP tools."""

from typing import Any, Protocol

from pydantic import Field

from .base import Arguments, conversation_tool


class McpSearchArguments(Arguments):
    server: str | None = Field(
        default=None,
        description="Optional MCP server name or slug to search (omit to search all connected servers).",
    )
    query: str = Field(
        default="",
        max_length=200,
        description="Optional keyword filter matching tool names or descriptions.",
    )


class McpCallArguments(Arguments):
    server: str = Field(
        min_length=1,
        max_length=100,
        description="MCP server slug or name from mcp_search_tools.",
    )
    tool: str = Field(
        min_length=1,
        max_length=200,
        description="Exact tool name from mcp_search_tools.",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON arguments matching the tool's schema.",
    )


class McpConversation(Protocol):
    async def search_mcp_tools(self, args: McpSearchArguments) -> dict: ...
    async def call_mcp_tool(self, args: McpCallArguments) -> dict: ...


@conversation_tool
async def mcp_search_tools(
    args: McpSearchArguments, conversation: McpConversation
) -> dict:
    """Search additional tools on connected MCP servers that are not directly bound."""
    return await conversation.search_mcp_tools(args)


@conversation_tool
async def mcp_call_tool(
    args: McpCallArguments, conversation: McpConversation
) -> dict:
    """Call a tool discovered via mcp_search_tools on a connected MCP server."""
    return await conversation.call_mcp_tool(args)


MCP_GATEWAY_TOOLS = [mcp_search_tools, mcp_call_tool]
