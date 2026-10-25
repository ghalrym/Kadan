"""Every tool the agent has, bound to the model with `bind_tools`."""

from langchain_core.tools import StructuredTool

from . import (
    bash,
    browser,
    code,
    conversation,
    display,
    file,
    jupyter,
    mcp,
    memory,
    nodejs,
    proxy,
    sandbox,
    shell,
    skills,
    util,
    workspace,
)

CONVERSATION_TOOLS: list[StructuredTool] = [*conversation.TOOLS, *workspace.TOOLS]
MEMORY_TOOLS: list[StructuredTool] = memory.TOOLS
COMPUTER_TOOLS: list[StructuredTool] = [
    *browser.TOOLS,
    *shell.TOOLS,
    *bash.TOOLS,
    *file.TOOLS,
    *code.TOOLS,
    *jupyter.TOOLS,
    *nodejs.TOOLS,
    *mcp.TOOLS,
    *sandbox.TOOLS,
    *proxy.TOOLS,
    *skills.TOOLS,
    *display.TOOLS,
    *util.TOOLS,
]
ALL_TOOLS = [*CONVERSATION_TOOLS, *MEMORY_TOOLS, *COMPUTER_TOOLS]
TOOLS_BY_NAME = {tool.name: tool for tool in ALL_TOOLS}


def request_tools(
    sandbox_configured: bool, has_references: bool, has_workflow: bool
) -> list[StructuredTool]:
    """Tools available for a request's selected data and configured computer."""
    tools = ALL_TOOLS if sandbox_configured else [*CONVERSATION_TOOLS, *MEMORY_TOOLS]
    return [
        tool
        for tool in tools
        if (tool is not conversation.read_request_references or has_references)
        and (tool is not conversation.read_workflow or has_workflow)
    ]
