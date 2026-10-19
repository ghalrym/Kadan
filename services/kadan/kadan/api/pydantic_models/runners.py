from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from .resources import AgentKey


class Runner(BaseModel):
    runner_id: AgentKey
    worker_id: UUID
    capacity: int = Field(default=5, ge=1, strict=True)
    lease_expires_at: datetime
    agent_keys: list[AgentKey] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_assignment(self):
        if self.lease_expires_at.utcoffset() is None:
            raise ValueError("Runner lease must have a timezone")
        if len(set(self.agent_keys)) != len(self.agent_keys):
            raise ValueError("Runner agent assignments must be unique")
        if len(self.agent_keys) > self.capacity:
            raise ValueError("Runner assignment exceeds capacity")
        return self
