"""Conversation tools bound to the agent model."""

from langchain_core.tools import StructuredTool

from . import conversation

CONVERSATION_TOOLS: list[StructuredTool] = conversation.TOOLS
TOOLS_BY_NAME = {tool.name: tool for tool in CONVERSATION_TOOLS}
