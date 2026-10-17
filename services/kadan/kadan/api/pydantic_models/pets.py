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
    sprite_version: Literal[2] = 2
    atlas_asset_id: UUID
    base_asset_id: UUID
    cell_width: Literal[192] = 192
    cell_height: Literal[208] = 208
    columns: Literal[8] = 8
    rows: Literal[11] = 11
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


class PetState(BaseModel):
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

    @model_validator(mode="after")
    def valid_pending_seed(self) -> Self:
        if self.pending_seed is not None and self.pending_request_id is None:
            raise ValueError("Pending seed requires a pending request")
        return self


class PetJob(PetState):
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
