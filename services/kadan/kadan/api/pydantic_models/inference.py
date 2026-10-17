from typing import Annotated, Literal

from pydantic import BaseModel, Field

from .resources import Message, Resource

MAX_AVATAR_BYTES = 20 * 1024 * 1024


class GeneratedImageSet(BaseModel):
    mime_type: Literal["image/png"]
    images_base64: list[
        Annotated[str, Field(max_length=4 * ((MAX_AVATAR_BYTES + 2) // 3))]
    ] = Field(min_length=1, max_length=1)


class AvatarGeneration(BaseModel):
    image: GeneratedImageSet


class Checkpoint(BaseModel):
    id: str
    status: Literal[
        "not_downloaded", "downloading", "cancelling", "cancelled", "failed", "complete"
    ]
    inference_available: bool


class TierModel(BaseModel):
    modality: str
    model: Checkpoint | None


class HardwareTier(BaseModel):
    id: str
    models: list[TierModel]


class HikariSettings(BaseModel):
    tier: str | None
    tiers: list[HardwareTier]


class CompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class PreparedChat(BaseModel):
    conversation: Resource
    user_message: Message
    messages: list[CompletionMessage]
    model_id: str
