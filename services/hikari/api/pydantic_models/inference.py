"""Inference capabilities and progress visible to API clients."""

from typing import Literal

from pydantic import BaseModel, Field

Feature = Literal[
    "llm",
    "image",
    "video",
    "stt",
    "audio_call",
    "stt_formatting",
    "tts",
    "decisions",
    "models",
    "tier_selection",
    "model_context",
    "whisper_selection",
    "whisper_formatting",
]
Operation = Literal["generate", "completion", "load", "unload", "download", "configure"]


class ImageProgress(BaseModel):
    phase: Literal["queued", "loading", "prompt", "denoise", "decode"] = "queued"
    image_index: int = Field(ge=1)
    image_count: int = Field(ge=1)
    steps_completed: int = Field(default=0, ge=0)
    steps_total: int = Field(ge=0)


class InferenceProgress(BaseModel):
    job_id: str
    workload: str
    stage: str
    value: int
    observed_unix_ns: int
    image: ImageProgress | None = None
