from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from .resources import AgentKey, SourceLink, Title


class Asset(BaseModel):
    id: UUID
    agent_key: AgentKey
    title: Title
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    is_image: bool
    version: int
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    sources: list[SourceLink] = Field(default_factory=list)
