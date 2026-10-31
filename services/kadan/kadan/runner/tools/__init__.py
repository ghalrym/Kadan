"""Every tool the agent has, bound to the model with `bind_tools`."""

from langchain_core.tools import StructuredTool

from . import (
    bash,
    browser,
    code,
    coding,
    conversation,
    display,
    file,
    history,
    jupyter,
    mcp,
    memory,
    nodejs,
    persona,
    proxy,
    sandbox,
    shell,
    skills,
    util,
    vision,
    workspace,
)

CONVERSATION_TOOLS: list[StructuredTool] = [
    *conversation.TOOLS,
    *workspace.TOOLS,
    *history.TOOLS,
]
MEMORY_TOOLS: list[StructuredTool] = memory.TOOLS
CODING_TOOLS: list[StructuredTool] = coding.START_TOOLS
PERSONA_TOOLS: list[StructuredTool] = persona.TOOLS
COMPUTER_TOOLS: list[StructuredTool] = [
    *coding.TOOLS,
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
ALL_TOOLS = [
    *CONVERSATION_TOOLS,
    *MEMORY_TOOLS,
    *PERSONA_TOOLS,
    *COMPUTER_TOOLS,
    *CODING_TOOLS,
]
VISION_TOOLS: list[StructuredTool] = vision.TOOLS
TOOLS_BY_NAME = {tool.name: tool for tool in [*ALL_TOOLS, *VISION_TOOLS]}
