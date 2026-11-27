"""Every tool the conversation agent has, bound with `bind_tools`."""

from langchain_core.tools import StructuredTool

from . import conversation, group_chat, history, memory, persona, updates, workspace

CONVERSATION_TOOLS: list[StructuredTool] = [
    *conversation.TOOLS,
    *workspace.TOOLS,
    *history.TOOLS,
    *updates.TOOLS,
]
GROUP_CHAT_TOOLS: list[StructuredTool] = group_chat.TOOLS
GROUP_CHAT_TOOLS_BY_NAME = {tool.name: tool for tool in GROUP_CHAT_TOOLS}
MEMORY_TOOLS: list[StructuredTool] = memory.TOOLS
PERSONA_TOOLS: list[StructuredTool] = persona.TOOLS
ALL_TOOLS = [
    *CONVERSATION_TOOLS,
    *MEMORY_TOOLS,
    *PERSONA_TOOLS,
]
TOOLS_BY_NAME = {tool.name: tool for tool in ALL_TOOLS}
