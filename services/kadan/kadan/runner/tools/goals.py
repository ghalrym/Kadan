"""The conversation assistant owns the goal beyond a single response."""

from datetime import datetime
from typing import Literal, Protocol, Self

from pydantic import Field, model_validator

from .base import Arguments, NoArguments, conversation_tool


class CheckpointTaskArguments(Arguments):
    expected_version: int | None = Field(default=None, ge=1, strict=True)
    goal: str = Field(min_length=1, max_length=4000, pattern=r"\S")
    progress: str = Field(default="", max_length=8000)
    next_step: str = Field(default="", max_length=4000)
    blocker: str | None = Field(default=None, max_length=4000)
    status: Literal["active", "waiting", "completed", "failed"]
    wait_kind: Literal["user", "tool", "timer"] | None = None
    wake_at: datetime | None = None
    message: str | None = Field(
        default=None, min_length=1, max_length=16000, pattern=r"\S"
    )

    @model_validator(mode="after")
    def valid_wait(self) -> Self:
        if (self.status == "waiting") != (self.wait_kind is not None):
            raise ValueError("Only waiting goals require a wait kind")
        if (self.wait_kind == "timer") != (self.wake_at is not None):
            raise ValueError("Only timer waits require a wake time")
        if self.wake_at is not None and self.wake_at.utcoffset() is None:
            raise ValueError("Wake time requires a timezone")
        if self.wait_kind == "user" and (not self.message or not self.blocker):
            raise ValueError(
                "Waiting for the user requires a visible question and blocker"
            )
        if self.status in {"active", "waiting"} and not self.next_step.strip():
            raise ValueError("Unfinished goals require a next step")
        if self.status == "failed" and not (self.blocker or "").strip():
            raise ValueError("Failed goals require an explanation")
        return self


class Goals(Protocol):
    async def read_task(self) -> dict: ...

    async def checkpoint_task(self, args: CheckpointTaskArguments) -> dict: ...


@conversation_tool
async def read_task(args: NoArguments, conversation: Goals) -> dict:
    """Read this conversation's saved goal, progress, next step and blockers.

    Read before updating; use the returned version. A finished response does not
    finish the goal. Paused or cancelled goals must not be continued.
    """
    return await conversation.read_task()


@conversation_tool
async def checkpoint_task(args: CheckpointTaskArguments, conversation: Goals) -> dict:
    """Create or checkpoint the ongoing goal this run advances.

    Use active to save progress and keep working now. Waiting, completed or failed
    ends this response. For user input, include the question as message and wait;
    never poll with inference. For a future check, use a timezone-aware wake_at at
    least thirty seconds away. Tool waits require actual submitted work. Include
    only verified progress; completed means the whole goal has been achieved.
    One unfinished goal is allowed per conversation. Null expected_version creates
    a goal; updates require the version returned by read_task.
    """
    return await conversation.checkpoint_task(args)


TOOLS = [read_task, checkpoint_task]
