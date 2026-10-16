"""Model metadata and status observed from the inference worker."""

from typing import Literal

from pydantic import BaseModel

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
        "llm", "video", "speech", "transcription", "formatting", "image", "decision"
    ]
    display_name: str | None
    license_url: str | None
    license_notice: str | None
    inference_available: bool
    context_limit: int | None
    architecture_context_limit: int | None


class ModelStoreStatus(BaseModel):
    models: list[CheckpointStatus]
    selected_model_id: str | None = None
    selected_whisper: str | None = None
    whisper_formatting: bool = True
    selected_image: str | None = None
