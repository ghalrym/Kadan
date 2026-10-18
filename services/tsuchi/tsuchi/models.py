from datetime import UTC, datetime
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
    return datetime.now(UTC)


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
    selected_pet_id: UUID | None = None


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
        return value.astimezone(UTC)


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


PET_ANIMATIONS = (
    "idle",
    "running-right",
    "running-left",
    "waving",
    "jumping",
    "failed",
    "waiting",
    "running",
    "review",
)
PET_FRAME_COUNTS = {
    "idle": 6,
    "running-right": 8,
    "running-left": 8,
    "waving": 4,
    "jumping": 5,
    "failed": 8,
    "waiting": 6,
    "running": 6,
    "review": 6,
    "look-0": 8,
    "look-1": 8,
}
PET_STEP_NAMES = (
    "base",
    *(
        f"{name}/{frame}"
        for name, count in PET_FRAME_COUNTS.items()
        for frame in range(count)
    ),
)
PetStatus = Literal[
    "queued", "running", "cancelling", "cancelled", "interrupted", "failed", "ready"
]


class PetAnimation(Model):
    name: str
    row: int = Field(ge=0, le=8)
    durations: list[Annotated[int, Field(ge=16, le=10000)]] = Field(
        min_length=1, max_length=8
    )


class PetManifest(Model):
    schema_version: Literal[1] = 1
    sprite_version: Literal[2] = 2
    atlas_asset_id: UUID
    base_asset_id: UUID
    cell_width: Literal[192] = 192
    cell_height: Literal[208] = 208
    columns: Literal[8] = 8
    rows: Literal[11] = 11
    pixelated: bool = False
    animations: list[PetAnimation] = Field(min_length=9, max_length=9)

    @model_validator(mode="after")
    def canonical_rows(self) -> Self:
        if [(item.name, item.row) for item in self.animations] != list(
            zip(PET_ANIMATIONS, range(9), strict=True)
        ):
            raise ValueError("Pet animations must use the nine canonical atlas rows")
        if self.atlas_asset_id == self.base_asset_id:
            raise ValueError("Pet atlas and base artwork must be separate assets")
        return self


class PetStep(Model):
    method: Literal["generated", "canonical"] = "generated"
    name: str = Field(min_length=1, max_length=100)
    source_asset_id: UUID | None = None
    output_asset_id: UUID
    raw_asset_id: UUID
    provenance_asset_id: UUID
    prompt: str = Field(min_length=1, max_length=20000)
    request_id: UUID
    model_id: str = Field(min_length=1, max_length=500)
    seed: int | None = None
    strength: float | None = Field(default=None, ge=0, le=1)
    meta: str = Field(default="", max_length=2000000)


class PetRejection(Model):
    step: PetStep
    reason: str = Field(min_length=1, max_length=10000)

    @field_validator("reason")
    @classmethod
    def nonblank_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Describe why the frame needs repair")
        return value.strip()


class PetPendingFrame(Model):
    name: str
    request_id: UUID
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


class PetState(Model):
    status: PetStatus = "queued"
    stage: str = Field(default="queued", min_length=1, max_length=500)
    completed_steps: int = Field(default=0, ge=0, le=74)
    total_steps: Literal[74] = 74
    error: str | None = Field(default=None, max_length=10000)
    cancel_requested: bool = False
    steps: list[PetStep] = Field(default_factory=list, max_length=74)
    parent_pet_id: UUID | None = None
    rejections: list[PetRejection] = Field(default_factory=list)
    manifest: PetManifest | None = None
    pending_step_name: str | None = Field(default=None, max_length=100)
    pending_request_id: UUID | None = None
    pending_seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    # The primary pending fields identify the earliest unfinished frame. This list
    # contains only the other members of its bounded concurrent cohort.
    additional_pending_frames: list[PetPendingFrame] = Field(
        default_factory=list, max_length=31
    )
    owner_id: UUID | None = None
    lease_expires_at: datetime | None = None

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

    @field_validator("lease_expires_at")
    @classmethod
    def aware_lease(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("Lease expiry requires a timezone")
        return value

    @model_validator(mode="after")
    def valid_progress(self) -> Self:
        if self.completed_steps != len(self.steps):
            raise ValueError("Completed steps must match persisted step outputs")
        names = [step.name for step in self.steps]
        if names != [name for name in PET_STEP_NAMES if name in names] or (
            names and names[0] != "base"
        ):
            raise ValueError("Pet steps must follow the canonical creation order")
        if len({step.output_asset_id for step in self.steps}) != len(self.steps):
            raise ValueError("Each pet step must have separate output artwork")
        if any(step.source_asset_id == step.output_asset_id for step in self.steps):
            raise ValueError("Source and output artwork must be separate assets")
        history = [*self.steps, *(rejection.step for rejection in self.rejections)]
        for step in history:
            if step.name not in PET_STEP_NAMES:
                raise ValueError("Pet history must reference canonical frames")
            if step.method == "canonical" and (
                step.name not in {"idle/0", "idle/5"}
                or not self.steps
                or step.source_asset_id != self.steps[0].output_asset_id
                or step.output_asset_id == step.source_asset_id
            ):
                raise ValueError(
                    "Canonical frames must copy the base into idle/0 or idle/5"
                )
        image_ids = {step.output_asset_id for step in history}
        image_ids.update(step.raw_asset_id for step in history)
        image_ids.update(
            step.source_asset_id for step in history if step.source_asset_id
        )
        provenance_ids = {step.provenance_asset_id for step in history}
        if image_ids & provenance_ids:
            raise ValueError("Provenance and artwork must be separate assets")
        if len({step.provenance_asset_id for step in self.steps}) != len(self.steps):
            raise ValueError("Each pet step must have separate provenance")
        if any(step.raw_asset_id == step.output_asset_id for step in self.steps):
            raise ValueError("Raw and prepared artwork must be separate assets")
        if (self.pending_step_name is None) != (self.pending_request_id is None):
            raise ValueError(
                "Pending step name and request ID must be supplied together"
            )
        if self.pending_seed is not None and self.pending_request_id is None:
            raise ValueError("Pending seed requires a pending request")
        if self.pending_step_name is not None:
            next_step = next(
                (name for name in PET_STEP_NAMES if name not in names), "atlas"
            )
            if self.pending_step_name != next_step:
                raise ValueError("Pending step must be the next unfinished step")
        if self.additional_pending_frames and self.pending_request_id is None:
            raise ValueError("Additional pending frames require a primary request")
        pending = self.pending_images
        pending_names = [frame.name for frame in pending]
        if pending_names != [
            name
            for name in PET_STEP_NAMES
            if name in pending_names and name not in names
        ] or len({frame.request_id for frame in pending}) != len(pending):
            raise ValueError(
                "Pending frames must be distinct unfinished frames in order"
            )
        if self.additional_pending_frames and (
            not self.steps
            or "base" in pending_names
            or any(name in {"idle/0", "idle/5"} for name in pending_names)
        ):
            raise ValueError("Only independent generated poses can share a cohort")
        if self.status in {"running", "cancelling"}:
            if self.owner_id is None or self.lease_expires_at is None:
                raise ValueError("Running jobs require an owner and lease")
        elif self.owner_id is not None or self.lease_expires_at is not None:
            raise ValueError("Only running jobs can retain an owner or lease")
        if self.status in {"cancelling", "cancelled"} and not self.cancel_requested:
            raise ValueError("Cancellation states require a cancellation request")
        if self.status == "ready":
            if (
                self.cancel_requested
                or self.manifest is None
                or len(self.steps) != len(PET_STEP_NAMES)
                or self.pending_step_name is not None
            ):
                raise ValueError("Ready pets require every step and an atlas manifest")
            if self.manifest.base_asset_id != self.steps[0].output_asset_id:
                raise ValueError("Manifest must reference the generated base artwork")
            if self.manifest.atlas_asset_id in image_ids | provenance_ids:
                raise ValueError("Atlas must be assembled into a separate asset")
        elif self.manifest is not None:
            raise ValueError("Only ready pets can contain an atlas manifest")
        return self

    def asset_references(self) -> set[UUID]:
        history = [*self.steps, *(rejection.step for rejection in self.rejections)]
        refs = {step.output_asset_id for step in history}
        refs.update(step.raw_asset_id for step in history)
        refs.update(step.provenance_asset_id for step in history)
        refs.update(step.source_asset_id for step in history if step.source_asset_id)
        if self.manifest:
            refs.update((self.manifest.atlas_asset_id, self.manifest.base_asset_id))
        return refs


class PetJob(PetState):
    id: UUID
    agent_key: AgentKey
    prompt: str = Field(min_length=1, max_length=10000)
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime


class ExportManifest(Model):
    schema_version: Literal[1] = 1
    exported_at: datetime
    agent: Agent
    resources: list[Resource] = Field(default_factory=list)
    versions: list[ExportRevision] = Field(default_factory=list)
    assets: list[Asset] = Field(default_factory=list)
    activity: list[Activity] = Field(default_factory=list)
    pet_jobs: list[PetJob] = Field(default_factory=list)
