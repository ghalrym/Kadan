from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CodexLogin(BaseModel):
    url: str
    code: str


class CodexSetup(BaseModel):
    state: Literal["signed_in", "signing_in", "signed_out", "missing", "unavailable"]
    message: str
    login: CodexLogin | None = None


class CodexThreadSummary(BaseModel):
    id: UUID
    title: str
    cwd: str
    created_at: datetime | None = None
    updated_at: datetime
    running: bool


class CodexThreadItem(BaseModel):
    kind: Literal["user", "assistant", "command"]
    text: str


class CodexThread(CodexThreadSummary):
    items: list[CodexThreadItem]
    items_truncated: bool = False


class CodexStatus(BaseModel):
    threads: list[CodexThreadSummary]


class CodexTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=16000)
    cwd: str | None = Field(
        default=None,
        pattern=r"^/",
        max_length=4096,
        description="Absolute project folder for a new thread.",
    )
    thread_id: UUID | None = Field(
        default=None,
        description="Continue this existing thread in its own folder instead of starting a new one.",
    )


class CodexStarted(BaseModel):
    thread_id: UUID
    cwd: str
    running: bool


class CodexOrigin(BaseModel):
    """The conversation run that started a Codex run and receives its result."""

    agent_key: str
    conversation_id: UUID
    run_id: UUID


class CodexFinished(BaseModel):
    request_id: UUID
    thread_id: UUID
    cwd: str
    origin: CodexOrigin | None
    status: Literal["completed", "failed", "stopped"]
    message: str


class CodexFinishedRuns(BaseModel):
    runs: list[CodexFinished]


class CodexStopped(BaseModel):
    thread_id: UUID
    running: bool
