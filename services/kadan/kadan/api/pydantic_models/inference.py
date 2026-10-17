from typing import Literal

from pydantic import BaseModel

from .resources import Message, Resource


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
