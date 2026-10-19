"""Persisted pet candidates and their generation provenance."""

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .resources import AgentKey


class PetAnimation(BaseModel):
    name: str
    row: int
    durations: list[int]


class PetManifest(BaseModel):
    schema_version: Literal[1] = 1
    sprite_version: Literal[1, 2] = 2
    atlas_asset_id: UUID
    # Imported atlases have no generated base artwork.
    base_asset_id: UUID | None = None
    cell_width: Literal[192] = 192
    cell_height: Literal[208] = 208
    columns: Literal[8] = 8
    rows: Literal[9, 11] = 11
    pixelated: bool = False
    animations: list[PetAnimation]


class PetStep(BaseModel):
    name: str
    method: Literal["generated", "canonical"] = "generated"
    source_asset_id: UUID | None = None
    output_asset_id: UUID
    raw_asset_id: UUID
    provenance_asset_id: UUID
    prompt: str
    request_id: UUID
    model_id: str
    seed: int | None = None
    strength: float | None = None
    meta: str = ""


class PetRejection(BaseModel):
    step: PetStep
    reason: str


class PetPendingFrame(BaseModel):
    name: str
    request_id: UUID
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


class PetState(BaseModel):
    origin: Literal["generated", "imported"] = "generated"
    status: Literal[
        "queued", "running", "cancelling", "cancelled", "interrupted", "failed", "ready"
    ] = "queued"
    stage: str = "queued"
    completed_steps: int = 0
    total_steps: Literal[74] = 74
    error: str | None = None
    cancel_requested: bool = False
    steps: list[PetStep] = Field(default_factory=list)
    rejections: list[PetRejection] = Field(default_factory=list)
    parent_pet_id: UUID | None = None
    manifest: PetManifest | None = None
    owner_id: UUID | None = None
    lease_expires_at: datetime | None = None
    pending_step_name: str | None = None
    pending_request_id: UUID | None = None
    pending_seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    # The primary pending fields identify the earliest unfinished frame. This list
    # contains only the other members of its bounded concurrent cohort.
    additional_pending_frames: list[PetPendingFrame] = Field(
        default_factory=list, max_length=31
    )

    @property
    def pending_images(self) -> list[PetPendingFrame]:
        primary = (
            [
                PetPendingFrame(
                    name=self.pending_step_name,
                    request_id=self.pending_request_id,
                    seed=self.pending_seed,
                )
            ]
            if self.pending_request_id is not None
            else []
        )
        return [*primary, *self.additional_pending_frames]

    @model_validator(mode="after")
    def valid_pending_frames(self) -> Self:
        if (self.pending_step_name is None) != (self.pending_request_id is None):
            raise ValueError(
                "Pending step name and request ID must be supplied together"
            )
        if self.pending_seed is not None and self.pending_request_id is None:
            raise ValueError("Pending seed requires a pending request")
        if self.additional_pending_frames and self.pending_request_id is None:
            raise ValueError("Additional pending frames require a primary request")
        pending = self.pending_images
        if len({frame.name for frame in pending}) != len(pending) or len(
            {frame.request_id for frame in pending}
        ) != len(pending):
            raise ValueError("Pending frames and request IDs must be unique")
        return self


class PetJob(PetState):
    trace_parent: str | None = Field(
        default=None, pattern=r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$"
    )
    id: UUID
    agent_key: AgentKey
    prompt: str
    version: int
    created_at: datetime
    updated_at: datetime


class PetAsset(BaseModel):
    id: UUID
    agent_key: AgentKey
    media_type: str
    sha256: str
    is_image: bool
    deleted_at: datetime | None = None


class GeneratedImage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model_id: str
    seeds: list[int] = Field(min_length=1, max_length=1)
    meta: str = ""
    images_base64: list[str] = Field(min_length=1, max_length=1)
    mime_type: Literal["image/png"]
    # Scheduling advice is transient, never part of saved image provenance.
    next_batch_capacity: int | None = Field(default=None, ge=1, le=32, exclude=True)
