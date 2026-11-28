"""Pydantic models for Kadan Group Chat rooms, participants, messages, and MCP Events."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from .pets import PetRef

GroupChatParticipantKind = Literal["kadan_agent", "external_agent"]
GroupChatSenderKind = Literal["user", "kadan_agent", "external_agent"]
GroupChatEventName = Literal["group_chat.turn_requested"]
GroupChatSubscriptionStatus = Literal["active", "expired", "failed", "revoked"]
GroupChatDeliveryStatus = Literal["pending", "delivered", "failed", "dropped"]
GroupChatPushStatus = Literal[
    "built_in", "active", "idle", "paused", "revoked", "expired", "error"
]


class GroupChatRoom(BaseModel):
    id: UUID
    title: str
    consecutive_agent_turns: int = Field(default=0, ge=0)
    max_consecutive_agent_turns: int = Field(default=6, ge=1, le=100)
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime


class GroupChatParticipant(BaseModel):
    """Tsuchi participant record (never contains raw tokens or hashes)."""

    id: UUID
    room_id: UUID
    handle: str
    display_name: str
    kind: GroupChatParticipantKind
    agent_key: str | None = None
    description: str = ""
    enabled: bool = True
    revoked_at: datetime | None = None
    last_seen_at: datetime | None = None
    cursor_seq: int = Field(default=0, ge=0)
    pet: PetRef | None = None
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime


class GroupChatMessage(BaseModel):
    id: UUID
    room_id: UUID
    seq: int = Field(ge=1)
    sender_kind: GroupChatSenderKind
    participant_id: UUID | None = None
    sender_name: str
    content: str
    handoff_to_participant_ids: list[UUID] = Field(default_factory=list)
    handoff_suppressed: bool = False
    reply_to_message_id: UUID | None = None
    client_request_id: str | None = None
    created_at: datetime


class GroupChatMessageWriteResult(BaseModel):
    message: GroupChatMessage
    room: GroupChatRoom
    effective_handoff_to_participant_ids: list[UUID] = Field(default_factory=list)
    turn_cap_reached: bool = False
    handoff_suppressed: bool = False
    idempotent_replay: bool = False
    enqueued_delivery_ids: list[UUID] = Field(default_factory=list)


class GroupChatMessagesPage(BaseModel):
    room: GroupChatRoom
    messages: list[GroupChatMessage]
    latest_seq: int = Field(ge=0)
    has_more: bool = False


class GroupChatEventSubscription(BaseModel):
    """Internal Tsuchi subscription record including signing secret."""

    id: UUID
    room_id: UUID
    participant_id: UUID
    event_name: GroupChatEventName = "group_chat.turn_requested"
    filters: dict[str, JsonValue] = Field(default_factory=dict)
    callback_url: str
    signing_secret: str
    status: GroupChatSubscriptionStatus = "active"
    expires_at: datetime
    last_delivered_seq: int = Field(default=0, ge=0)
    last_error: str | None = None
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime


class GroupChatWebhookDelivery(BaseModel):
    id: UUID
    subscription_id: UUID
    room_id: UUID
    participant_id: UUID
    seq: int = Field(ge=1)
    message_id: UUID
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    attempts: int = Field(default=0, ge=0)
    next_attempt_at: datetime
    status: GroupChatDeliveryStatus = "pending"
    last_error: str | None = None
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime


class GroupChatParticipantPublic(BaseModel):
    """Browser-facing participant view with redacted secrets and plain-language status."""

    id: UUID
    room_id: UUID
    handle: str
    display_name: str
    kind: GroupChatParticipantKind
    agent_key: str | None = None
    description: str = ""
    enabled: bool = True
    revoked_at: datetime | None = None
    last_seen_at: datetime | None = None
    cursor_seq: int = Field(default=0, ge=0)
    pet: PetRef | None = None
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime
    push_status: GroupChatPushStatus
    push_status_detail: str
    active_subscriptions: int = Field(default=0, ge=0)


class GroupChatRoomSnapshot(BaseModel):
    room: GroupChatRoom
    participants: list[GroupChatParticipantPublic]
    messages: list[GroupChatMessage]
    latest_seq: int = Field(ge=0)
    has_more: bool = False
    turn_cap_reached: bool = False
    mcp_url: str


class GroupChatCreateExternalParticipantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=500)
    handle: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_-]{0,47}$")

    @field_validator("display_name")
    @classmethod
    def nonblank_display_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Display name cannot be blank")
        return stripped

    @field_validator("description")
    @classmethod
    def trim_description(cls, value: str) -> str:
        return value.strip()


class GroupChatCreateExternalParticipantResponse(BaseModel):
    participant: GroupChatParticipantPublic
    bearer_token: str
    mcp_url: str


class GroupChatUpdateParticipantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=500)
    enabled: bool | None = None
    rotate_token: bool = False
    revoke: bool = False
    pet: PetRef | None = None
    clear_pet: bool = False
    expected_version: int = Field(ge=1)

    @field_validator("display_name")
    @classmethod
    def nonblank_update_display_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("Display name cannot be blank")
        return stripped

    @field_validator("description")
    @classmethod
    def trim_update_description(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @model_validator(mode="after")
    def one_pet_change(self) -> GroupChatUpdateParticipantRequest:
        if self.pet is not None and self.clear_pet:
            raise ValueError("Choose a pet or clear it, not both")
        return self


class GroupChatUpdateParticipantResponse(BaseModel):
    participant: GroupChatParticipantPublic
    bearer_token: str | None = None
    mcp_url: str | None = None


class GroupChatRoomUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=200)
    max_consecutive_agent_turns: int | None = Field(default=None, ge=1, le=100)
    expected_version: int = Field(ge=1)


class GroupChatUserMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=16000)
    handoff_to_participant_ids: list[UUID] = Field(default_factory=list, max_length=20)
    reply_to_message_id: UUID | None = None
    client_request_id: str | None = Field(default=None, max_length=256)

    @field_validator("content")
    @classmethod
    def nonblank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message content cannot be blank")
        return value


class GroupChatUserMessageResponse(BaseModel):
    message: GroupChatMessage
    room: GroupChatRoom
    effective_handoff_to_participant_ids: list[UUID] = Field(default_factory=list)
    turn_cap_reached: bool = False
    handoff_suppressed: bool = False
    idempotent_replay: bool = False
