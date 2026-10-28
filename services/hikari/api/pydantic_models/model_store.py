"""Model metadata and status observed from the inference worker."""

from typing import Literal

from pydantic import BaseModel, Field

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
        "decision",
        "vision",
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
    """Fixed assignments; checkpoint availability is observed from workers."""

    llm: Literal["lfm3-1.7b-bf16"] = "lfm3-1.7b-bf16"
    vision: Literal["lfm3-vl-2b-bf16"] = "lfm3-vl-2b-bf16"
    speech: Literal["lfm3-audio-2b-bf16"] = "lfm3-audio-2b-bf16"
    image: Literal["qwen-image-2.1-turbo"] = "qwen-image-2.1-turbo"
    video: Literal["h3-fl2va-int8-turbo"] = "h3-fl2va-int8-turbo"
    transcription: Literal["whisper-medium"] = "whisper-medium"
    formatting: Literal["lfm3-1.7b-bf16"] = "lfm3-1.7b-bf16"
    decisions: Literal["laya"] = "laya"


class VideoDefaults(BaseModel):
    """Suggested clip settings; request limits and memory admission are separate."""

    duration: int = Field(default=5, ge=1, le=15)
    resolution: Literal["480p", "768p"] = "480p"


class ModelStoreStatus(BaseModel):
    models: list[CheckpointStatus]
    defaults: ModelAssignments = Field(default_factory=ModelAssignments)
    video_defaults: VideoDefaults = Field(default_factory=VideoDefaults)
    selected_model_id: str | None = None
    selected_whisper: str | None = None
    whisper_formatting: bool = True
