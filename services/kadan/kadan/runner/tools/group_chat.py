"""Group Chat MCP tools bound for Kadan's built-in agent during Group Chat wake runs."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from pydantic import Field

from .base import Arguments, NoArguments, conversation_tool


class GroupChatReadMessagesArguments(Arguments):
    after_seq: int | None = Field(
        default=None,
        ge=0,
        description="Read messages strictly newer than this sequence number.",
    )
    before_seq: int | None = Field(
        default=None,
        ge=1,
        description="Read messages strictly older than this sequence number.",
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Maximum number of messages to return (1-100).",
    )


class GroupChatSendMessageArguments(Arguments):
    content: str = Field(
        min_length=1,
        max_length=16000,
        pattern=r"\S",
        description="Text to post in the Group Chat room.",
    )
    handoff_to: str | list[str] | None = Field(
        default=None,
        description=(
            "Optional participant @handle or UUID (or list of them) to explicitly "
            "hand the turn to another agent. Omit to return the floor to the user. "
            "Self-handoff is rejected."
        ),
    )
    reply_to_message_id: UUID | None = Field(
        default=None,
        description="Optional message UUID you are replying to.",
    )


class GroupChatConversation(Protocol):
    async def group_chat_list_participants(self) -> dict: ...

    async def group_chat_read_messages(
        self, args: GroupChatReadMessagesArguments
    ) -> dict: ...

    async def group_chat_send_message(
        self, args: GroupChatSendMessageArguments
    ) -> dict: ...


@conversation_tool
async def group_chat_list_participants(
    args: NoArguments, conversation: GroupChatConversation
) -> dict:
    """List participants in Group Chat with their @handles, roles, and active status."""
    return await conversation.group_chat_list_participants()


@conversation_tool
async def group_chat_read_messages(
    args: GroupChatReadMessagesArguments, conversation: GroupChatConversation
) -> dict:
    """Read paginated Group Chat messages and current turn-cap status."""
    return await conversation.group_chat_read_messages(args)


@conversation_tool
async def group_chat_send_message(
    args: GroupChatSendMessageArguments, conversation: GroupChatConversation
) -> dict:
    """Post a reply to Group Chat. Pass handoff_to only when handing the turn to another agent."""
    return await conversation.group_chat_send_message(args)


TOOLS = [
    group_chat_list_participants,
    group_chat_read_messages,
    group_chat_send_message,
]
