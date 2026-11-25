from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from .resources import AgentKey


class ConversationSummary(BaseModel):
    """A chat as listed in Kadan; message bodies stay in the conversation resource."""

    id: UUID
    agent_key: AgentKey
    title: str
    # Set only when the chat's parent is a Space (a Tsuchi project).
    space_id: UUID | None
    message_count: int
    preview: str
    version: int
    created_at: datetime
    updated_at: datetime


class Space(BaseModel):
    """Kadan's name for a Tsuchi project: a title, instructions and its own chats."""

    id: UUID
    agent_key: AgentKey
    title: str
    instructions: str
    version: int
    created_at: datetime
    updated_at: datetime
