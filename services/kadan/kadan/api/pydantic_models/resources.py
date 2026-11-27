from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

AgentKey = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")]
Title = Annotated[str, Field(min_length=1, max_length=500)]


class Agent(BaseModel):
    key: AgentKey
    name: Title
    created_at: datetime | None = None
    selected_pet_id: UUID | None = None
    persona_id: UUID | None = None


class ResourceKind(StrEnum):
    PERSONA = "persona"
    MEMORY = "memory"
    SPACE = "space"
    DECK = "deck"
    CONVERSATION = "conversation"
    # Groups Space pages, slides, files and other folders; it has no content.
    FOLDER = "folder"


class SourceLink(BaseModel):
    label: str
    url: str


class Message(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    attachment_ids: list[UUID] = Field(default_factory=list)
    resource_ids: list[UUID] = Field(default_factory=list)
    sources: list[SourceLink] = Field(default_factory=list)


class Slide(BaseModel):
    id: UUID
    layout: str
    title: str
    body: str
    notes: str
    image_asset_id: UUID | None


class Deck(BaseModel):
    schema_version: int
    width: int
    height: int
    slides: list[Slide]


class Resource(BaseModel):
    id: UUID
    agent_key: AgentKey
    kind: ResourceKind
    title: str
    markdown: str = ""
    parent_id: UUID | None = None
    tags: list[str] = Field(default_factory=list)
    sources: list[SourceLink] = Field(default_factory=list)
    attachment_ids: list[UUID] = Field(default_factory=list)
    deck: Deck | None = None
    messages: list[Message] = Field(default_factory=list)
    compacted_summary: str | None = None
    compacted_through_message_id: UUID | None = None
    compacted_at: datetime | None = None
    memory_extracted_through_message_id: UUID | None = None
    core_memories_snapshot: str | None = None
    core_memories_refreshed_at: datetime | None = None
    pinned: bool = False
    version: int
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class WorkspaceDocument(Resource):
    kind: Literal[ResourceKind.SPACE, ResourceKind.DECK]


class Memory(Resource):
    kind: Literal[ResourceKind.MEMORY]
