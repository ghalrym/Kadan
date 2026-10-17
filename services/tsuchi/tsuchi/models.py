from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    TypeAdapter,
    field_validator,
    model_validator,
)

AgentKey = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")]
Title = Annotated[str, Field(min_length=1, max_length=500)]
MAX_CONVERSATION_MESSAGES = 10000


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class AgentInput(Model):
    key: AgentKey
    name: Title

    @field_validator("name")
    @classmethod
    def nonblank_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Name cannot be blank")
        return value.strip()


class Agent(AgentInput):
    created_at: datetime
    avatar_asset_id: UUID | None = None


class SourceLink(Model):
    label: str = Field(max_length=500)
    url: HttpUrl


class ResourceKind(StrEnum):
    PERSONA = "persona"
    MEMORY = "memory"
    SPACE = "space"
    DECK = "deck"
    CONVERSATION = "conversation"


class Slide(Model):
    id: UUID
    layout: Literal["title", "text", "split", "image"]
    title: str = Field(default="", max_length=2000)
    body: str = Field(default="", max_length=100000)
    notes: str = Field(default="", max_length=100000)
    image_asset_id: UUID | None = None


class Deck(Model):
    schema_version: Literal[1] = 1
    width: Literal[1920] = 1920
    height: Literal[1080] = 1080
    slides: list[Slide] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def unique_slide_ids(self) -> Self:
        if len({slide.id for slide in self.slides}) != len(self.slides):
            raise ValueError("Slide IDs must be unique")
        return self


class Message(Model):
    id: UUID
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(max_length=1000000)
    created_at: datetime
    attachment_ids: list[UUID] = Field(default_factory=list, max_length=100)
    resource_ids: list[UUID] = Field(default_factory=list, max_length=100)
    sources: list[SourceLink] = Field(default_factory=list, max_length=100)

    @field_validator("created_at")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Message timestamps require a timezone")
        return value.astimezone(timezone.utc)


class Metadata(Model):
    title: Title
    tags: list[Annotated[str, Field(min_length=1, max_length=100)]] = Field(
        default_factory=list, max_length=100
    )
    sources: list[SourceLink] = Field(default_factory=list, max_length=100)

    @field_validator("title")
    @classmethod
    def nonblank_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Title cannot be blank")
        return value.strip()

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, tags: list[str]) -> list[str]:
        return list(dict.fromkeys(tag.strip() for tag in tags if tag.strip()))


class ResourceContent(Metadata):
    markdown: str = Field(default="", max_length=2000000)
    parent_id: UUID | None = None
    attachment_ids: list[UUID] = Field(default_factory=list, max_length=100)


class MarkdownContent(ResourceContent):
    kind: Literal[ResourceKind.PERSONA, ResourceKind.MEMORY, ResourceKind.SPACE]
    deck: None = None
    messages: list[Message] = Field(default_factory=list, max_length=0)


class DeckContent(ResourceContent):
    kind: Literal[ResourceKind.DECK]
    deck: Deck
    messages: list[Message] = Field(default_factory=list, max_length=0)


class ConversationContent(ResourceContent):
    kind: Literal[ResourceKind.CONVERSATION]
    deck: None = None
    messages: list[Message] = Field(
        default_factory=list, max_length=MAX_CONVERSATION_MESSAGES
    )

    @model_validator(mode="after")
    def unique_message_ids(self) -> Self:
        if len({message.id for message in self.messages}) != len(self.messages):
            raise ValueError("Message IDs must be unique")
        return self


ResourceInput = Annotated[
    MarkdownContent | DeckContent | ConversationContent, Field(discriminator="kind")
]
resource_input_adapter = TypeAdapter(ResourceInput)


class VersionInput(Model):
    expected_version: int = Field(ge=1)


class MarkdownUpdate(MarkdownContent, VersionInput):
    pass


class DeckUpdate(DeckContent, VersionInput):
    pass


class ConversationUpdate(ConversationContent, VersionInput):
    pass


ResourceUpdate = Annotated[
    MarkdownUpdate | DeckUpdate | ConversationUpdate, Field(discriminator="kind")
]


class ResourceRecord(Model):
    id: UUID
    agent_key: AgentKey
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class MarkdownResource(MarkdownContent, ResourceRecord):
    pass


class DeckResource(DeckContent, ResourceRecord):
    pass


class ConversationResource(ConversationContent, ResourceRecord):
    pass


Resource = Annotated[
    MarkdownResource | DeckResource | ConversationResource, Field(discriminator="kind")
]
resource_adapter = TypeAdapter(Resource)


class ResourceVersion(Model):
    version: int = Field(ge=1)
    created_at: datetime
    snapshot: Resource


class MessageAppend(VersionInput):
    message: Message


class AssetUpdate(Metadata):
    expected_version: int = Field(ge=1)


class Asset(Metadata):
    id: UUID
    agent_key: AgentKey
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    is_image: bool
    version: int
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class Activity(Model):
    id: UUID
    action: str
    entity_type: Literal["agent", "resource", "asset"]
    entity_id: str
    title: str
    created_at: datetime


class Stats(Model):
    resources: dict[ResourceKind, int]
    assets: int
    images: int
    trashed_resources: int
    trashed_assets: int
    storage_bytes: int
    database_bytes: int
    revisions: int


class ExportRevision(ResourceVersion):
    resource_id: UUID

    @model_validator(mode="after")
    def matches_snapshot(self) -> Self:
        if (
            self.resource_id != self.snapshot.id
            or self.version != self.snapshot.version
        ):
            raise ValueError(
                "Export revision must identify its resource snapshot and version"
            )
        return self


class ExportManifest(Model):
    schema_version: Literal[1] = 1
    exported_at: datetime
    agent: Agent
    resources: list[Resource] = Field(default_factory=list)
    versions: list[ExportRevision] = Field(default_factory=list)
    assets: list[Asset] = Field(default_factory=list)
    activity: list[Activity] = Field(default_factory=list)
