"""Model metadata and status observed from the inference worker."""

from typing import Literal

from pydantic import BaseModel, Field

from api.pydantic_models.inference_task import LlmModel

DownloadStatus = Literal[
    "not_downloaded", "downloading", "cancelling", "cancelled", "failed", "complete"
]


class DownloadObservation(BaseModel):
    status: DownloadStatus = "downloading"
    downloaded_bytes: int = 0
    total_bytes: int = 0
    error: str | None = None


class CheckpointStatus(BaseModel):
    status: DownloadStatus
    downloaded_bytes: int
    total_bytes: int
    error: str | None
    id: str
    repo_id: str
    revision: str
    license: str
    estimated_bytes: int
    kind: Literal[
        "llm",
        "video",
        "speech",
        "transcription",
        "formatting",
        "image",
        "audio",
    ]
    display_name: str | None
    short_name: str | None = None
    license_url: str | None
    license_notice: str | None
    inference_available: bool
    context_limit: int | None
    architecture_context_limit: int | None


class ModelAssignments(BaseModel):
    """Fixed capability assignments, independent of machine resources."""

    llm: LlmModel = "inkling-2-mini-iq4-nl"
    audio_call: Literal["lfm3-audio-2b-bf16"] = "lfm3-audio-2b-bf16"
    image: Literal["flux.2-klein-4b"] = "flux.2-klein-4b"
    video: Literal["h3-fl2va-int8-turbo"] = "h3-fl2va-int8-turbo"
    speech: Literal["qwen-tts-1.7b-custom"] = "qwen-tts-1.7b-custom"
    transcription: Literal["whisper-medium"] = "whisper-medium"
    formatting: Literal["superwhisper-s1-mini"] = "superwhisper-s1-mini"
    decisions: LlmModel = "inkling-2-mini-iq4-nl"


class VideoDefaults(BaseModel):
    """Suggested clip settings; request limits and memory admission are separate."""

    duration: int = Field(default=5, ge=1, le=15)
    resolution: Literal["480p", "768p"] = "480p"


class ModelStoreStatus(BaseModel):
    models: list[CheckpointStatus]
    defaults: ModelAssignments = Field(default_factory=ModelAssignments)
    video_defaults: VideoDefaults = Field(default_factory=VideoDefaults)
    selected_model_id: LlmModel = "inkling-2-mini-iq4-nl"
    whisper_formatting: bool = True
