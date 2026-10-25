"""Every tool the agent has, bound to the model with `bind_tools`."""

from langchain_core.tools import StructuredTool

from . import (
    bash,
    browser,
    code,
    conversation,
    display,
    file,
    history,
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

CONVERSATION_TOOLS: list[StructuredTool] = [
    *conversation.TOOLS,
    *workspace.TOOLS,
    *history.TOOLS,
]
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
