"""Every tool the conversation and computer agents have, bound with `bind_tools`."""

from langchain_core.tools import StructuredTool

from . import (
    bash,
    browser,
    code,
    coding,
    computer,
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
# Codex threads stay with the conversation agent; they run on the computer.
CODEX_TOOLS: list[StructuredTool] = coding.TOOLS
DELEGATION_TOOLS: list[StructuredTool] = computer.TOOLS
PERSONA_TOOLS: list[StructuredTool] = persona.TOOLS
# Only the computer agent binds these.
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
TASK_TOOLS: list[StructuredTool] = computer.TASK_TOOLS
VISION_TOOLS: list[StructuredTool] = vision.TOOLS
ALL_TOOLS = [
    *CONVERSATION_TOOLS,
    *MEMORY_TOOLS,
    *PERSONA_TOOLS,
    *CODING_TOOLS,
    *CODEX_TOOLS,
    *DELEGATION_TOOLS,
]
COMPUTER_AGENT_TOOLS = [*COMPUTER_TOOLS, *VISION_TOOLS, *TASK_TOOLS]
TOOLS_BY_NAME = {tool.name: tool for tool in [*ALL_TOOLS, *COMPUTER_AGENT_TOOLS]}
