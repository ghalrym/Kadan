"""Durable agent runs and image jobs owned by Tsuchi."""

from datetime import datetime
from typing import Annotated, Literal, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .resources import AgentKey


class RequestReference(BaseModel):
    resource_id: UUID
    expected_version: int = Field(ge=1, strict=True)


class ReferenceSnapshot(BaseModel):
    resource_id: UUID
    kind: Literal["space", "memory"]
    title: str = Field(min_length=1, max_length=500)
    version: int = Field(ge=1)
    markdown: str = Field(max_length=12000)


class RunMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=1000000)
    attachment_ids: list[UUID] = Field(default_factory=list, max_length=100)


class ToolCommand(BaseModel):
    """An action selected by the agent."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    tool: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_]+$")
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class RunCall(BaseModel):
    role: Literal["call"] = "call"
    call_id: str = Field(
        default_factory=lambda: uuid4().hex, min_length=1, max_length=120
    )
    command: ToolCommand


class RunResult(BaseModel):
    role: Literal["result"] = "result"
    call_id: str = Field(min_length=1, max_length=120)
    data: dict[str, JsonValue]


class RunObservation(BaseModel):
    role: Literal["observation", "rejection"] = "observation"
    name: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_]+$")
    data: dict[str, JsonValue] | list[JsonValue]


class MigratedMarker(BaseModel):
    """Occupies a retired scheduling row so saved side-effect identities stay valid."""

    role: Literal["retired"] = "retired"


class UnrecoveredRecord(BaseModel):
    """Original journal row whose safe replay could not be established."""

    role: Literal["unrecovered"] = "unrecovered"
    original: JsonValue


TranscriptEntry = Annotated[
    RunMessage
    | RunCall
    | RunResult
    | RunObservation
    | MigratedMarker
    | UnrecoveredRecord,
    Field(discriminator="role"),
]


class AgentRunState(BaseModel):
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = "queued"
    transcript: list[TranscriptEntry] = Field(default_factory=list, max_length=1000)
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


class RunEvent(BaseModel):
    # "computer" remains readable for events saved before the computer was removed.
    kind: Literal["codex", "computer", "t3"]
    content: str = Field(min_length=1, max_length=20000)


class AgentRun(AgentRunState):
    trace_parent: str | None = Field(
        default=None, pattern=r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$"
    )
    id: UUID
    agent_key: AgentKey
    conversation_id: UUID
    trigger_message_id: UUID | None = None
    trigger_job_id: UUID | None = None
    trigger_event: RunEvent | None = None
    persona_id: UUID | None = None
    references: list[ReferenceSnapshot] = Field(default_factory=list, max_length=3)
    delivered_message_ids: list[UUID] = Field(default_factory=list)
    draft: str | None = None
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime


class ImageJobState(BaseModel):
    status: Literal[
        "queued", "running", "completed", "failed", "cancelling", "cancelled"
    ] = "queued"
    upload_status: Literal["not_started", "pending", "completed", "failed"] = (
        "not_started"
    )
    owner_id: UUID | None = None
    lease_expires_at: datetime | None = None
    asset_id: UUID | None = None
    error: str | None = Field(default=None, max_length=10000)

    @model_validator(mode="after")
    def valid_result_and_lease(self) -> Self:
        if self.status in {"running", "cancelling"}:
            if self.owner_id is None or self.lease_expires_at is None:
                raise ValueError("Active image jobs require an owner and lease")
        elif self.owner_id is not None or self.lease_expires_at is not None:
            raise ValueError("Only active image jobs can retain an owner or lease")
        if (
            self.lease_expires_at is not None
            and self.lease_expires_at.utcoffset() is None
        ):
            raise ValueError("Lease timestamps require a timezone")
        if (self.status == "completed") != (self.asset_id is not None):
            raise ValueError("Only completed image jobs require an image asset")
        if self.upload_status == "pending" and self.status in {
            "completed",
            "failed",
            "cancelled",
        }:
            raise ValueError("An unresolved upload must remain recoverable")
        if self.status == "completed" and self.upload_status == "failed":
            raise ValueError("A failed upload cannot complete an image job")
        return self


class ImageJob(ImageJobState):
    trace_parent: str | None = Field(
        default=None, pattern=r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$"
    )
    id: UUID
    agent_key: AgentKey
    run_id: UUID
    conversation_id: UUID
    prompt: str = Field(min_length=1, max_length=10000)
    request_id: UUID
    event_run_id: UUID
    delivered_message_id: UUID | None = None
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime
