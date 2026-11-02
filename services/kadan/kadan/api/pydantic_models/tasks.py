"""Saved assistant goals, separate from the response that advances them."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .resources import AgentKey


class AssistantTask(BaseModel):
    id: UUID
    agent_key: AgentKey
    conversation_id: UUID
    origin_run_id: UUID
    current_run_id: UUID | None = None
    status: Literal["active", "waiting", "paused", "completed", "failed", "cancelled"]
    goal: str
    progress: str
    next_step: str
    blocker: str | None = None
    wait_kind: Literal["user", "tool", "timer"] | None = None
    wake_at: datetime | None = None
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime


class TaskControl(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_version: int = Field(ge=1, strict=True)
