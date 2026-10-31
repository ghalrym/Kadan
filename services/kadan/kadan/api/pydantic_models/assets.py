from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from .resources import AgentKey, SourceLink


class Asset(BaseModel):
    id: UUID
    agent_key: AgentKey
    title: str = Field(min_length=1, max_length=500)
    filename: str = Field(min_length=1, max_length=255)
    media_type: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(ge=0, strict=True)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    is_image: bool
    tags: list[str] = Field(default_factory=list)
    sources: list[SourceLink] = Field(default_factory=list)
    version: int = Field(ge=1, strict=True)
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None
