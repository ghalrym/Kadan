from typing import Literal

from pydantic import BaseModel

DownloadState = Literal[
    "not_downloaded", "downloading", "cancelling", "cancelled", "failed", "complete"
]


class Checkpoint(BaseModel):
    id: str
    status: DownloadState
    inference_available: bool
    # Tokens. The configured window, else the checkpoint's own.
    context_limit: int | None = None
    architecture_context_limit: int | None = None


class TierModel(BaseModel):
    modality: str
    model: Checkpoint | None


class HardwareTier(BaseModel):
    id: str
    models: list[TierModel]


class HikariSettings(BaseModel):
    tier: str | None
    tiers: list[HardwareTier]


class HikariModels(BaseModel):
    models: list[Checkpoint]


class HikariModelLifecycle(BaseModel):
    state: Literal["unloaded", "loading", "ready", "offloaded", "unloading", "error"]
    model_id: str | None = None


class CompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
