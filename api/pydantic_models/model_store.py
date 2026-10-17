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


TierId = Literal["go", "plus", "pro", "ultra"]
TierModality = Literal["llm", "image", "video", "speech", "transcription", "formatting"]


class HardwareTier(BaseModel):
    """A hardware profile fixing one checkpoint per modality; None is unassigned."""

    id: TierId
    name: str
    vram_gb: int
    ram_gb: int
    models: dict[TierModality, str | None]


class ModelStoreStatus(BaseModel):
    models: list[CheckpointStatus]
    tiers: list[HardwareTier] = []
    selected_tier: TierId | None = None
    selected_model_id: str | None = None
    selected_whisper: str | None = None
    whisper_formatting: bool = True
    selected_image: str | None = None
