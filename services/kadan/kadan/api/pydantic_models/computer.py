"""The computer agent's tasks and long-lived thread, owned by Tsuchi."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from .agent_runs import RunMessage
from .resources import AgentKey


class ComputerTaskOrigin(BaseModel):
    conversation_id: UUID
    run_id: UUID


class ComputerTask(BaseModel):
    id: UUID
    agent_key: AgentKey
    instructions: str = Field(min_length=1, max_length=10000)
    origin: ComputerTaskOrigin | None = None
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = "queued"
    summary: str | None = Field(default=None, max_length=10000)
    event_run_id: UUID | None = None
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime


class ComputerSteering(BaseModel):
    id: UUID
    content: str = Field(min_length=1, max_length=10000)
    created_at: datetime


class ComputerThread(BaseModel):
    agent_key: AgentKey
    transcript: list[RunMessage] = Field(default_factory=list, max_length=1000)
    owner_id: UUID | None = None
    lease_expires_at: datetime | None = None
    steering: list[ComputerSteering] = Field(default_factory=list)
    version: int = Field(ge=1)
    updated_at: datetime


class ComputerRequest(BaseModel):
    kind: Literal["steering", "task"]
    task: ComputerTask
