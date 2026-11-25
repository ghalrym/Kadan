from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from .resources import AgentKey


class ConversationSummary(BaseModel):
    """A chat as listed in Kadan; message bodies stay in the conversation resource."""

    id: UUID
    agent_key: AgentKey
    title: str
    message_count: int
    preview: str
    version: int
    created_at: datetime
    updated_at: datetime
