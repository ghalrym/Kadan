"""Display fields for conversation polling, separate from durable work records."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class AgentRunSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    conversation_id: UUID
    trigger_message_id: UUID | None
    trigger_job_id: UUID | None
    delivered_message_ids: list[UUID]
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    error: str | None
    created_at: datetime
    updated_at: datetime


class ImageJobSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    conversation_id: UUID
    prompt: str
    status: Literal[
        "queued", "running", "completed", "failed", "cancelling", "cancelled"
    ]
    error: str | None
    delivered_message_id: UUID | None
    created_at: datetime
    updated_at: datetime
