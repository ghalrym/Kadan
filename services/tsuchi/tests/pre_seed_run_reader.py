"""Frozen strict run reader before atomic initial-command enqueue.

Kept independent of current models to exercise stored-record compatibility.
"""

from datetime import datetime
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

AgentKey = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class RunMessage(Model):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=1000000)


class AgentRunState(Model):
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = "queued"
    transcript: list[RunMessage] = Field(default_factory=list, max_length=1000)
    owner_id: UUID | None = None
    lease_expires_at: datetime | None = None
    error: str | None = Field(default=None, max_length=10000)

    @model_validator(mode="after")
    def valid_lease(self) -> Self:
        if self.status == "running":
            if self.owner_id is None or self.lease_expires_at is None:
                raise ValueError("Running agent runs require an owner and lease")
        elif self.owner_id is not None or self.lease_expires_at is not None:
            raise ValueError("Only running agent runs can retain an owner or lease")
        if (
            self.lease_expires_at is not None
            and self.lease_expires_at.utcoffset() is None
        ):
            raise ValueError("Lease timestamps require a timezone")
        return self


class AgentRun(AgentRunState):
    trace_parent: str | None = Field(
        default=None, pattern=r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$"
    )
    id: UUID
    agent_key: AgentKey
    conversation_id: UUID
    trigger_message_id: UUID | None = None
    trigger_job_id: UUID | None = None
    persona_id: UUID | None = None
    delivered_message_ids: list[UUID] = Field(default_factory=list)
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime
