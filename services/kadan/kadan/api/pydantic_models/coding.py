from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CodingSetup(BaseModel):
    state: Literal["ready", "missing", "auth_required", "unavailable"]
    message: str
    version: str | None = None


class CodingEvent(BaseModel):
    sequence: int
    kind: str
    message: str


class CodingJob(BaseModel):
    id: UUID
    request_id: UUID
    status: Literal[
        "starting",
        "running",
        "stopping",
        "completed",
        "failed",
        "cancelled",
        "timed_out",
        "interrupted",
        "cleanup_required",
    ]
    cwd: str
    session_id: str | None = None
    resume_job_id: UUID | None = None
    created_at: str
    finished_at: str | None = None
    message: str
    events: list[CodingEvent] = Field(default_factory=list)
    next_cursor: int = 0
    events_truncated: bool = False


class CodingStatus(BaseModel):
    setup: CodingSetup
    jobs: list[CodingJob]


class CodingTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=16000)
    cwd: str = Field(pattern=r"^/", max_length=4096)
    timeout_seconds: int = Field(default=1800, ge=60, le=3600, strict=True)
    resume_job_id: UUID | None = None


class CodingStart(CodingTask):
    request_id: UUID
