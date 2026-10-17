"""Settings view of the worker-owned model store."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from api.pydantic_models.inference_task import WhisperFormattingTask
from api.pydantic_models.model_store import (
    CheckpointStatus,
    TierId,
    TierModality,
    VideoDefaults,
)
from api.services.inference import inference
from api.services.inference_errors import InferenceFailure

router = APIRouter(prefix="/v1/settings", tags=["Settings"])


class TierModel(BaseModel):
    """A tier's checkpoint for one modality, with its live download state."""

    modality: TierModality
    model: CheckpointStatus | None = Field(
        description="Null while the tier assigns no checkpoint to this modality."
    )


class TierSetting(BaseModel):
    """A selectable hardware tier and the checkpoints it downloads."""

    id: TierId
    name: str
    vram_gb: int
    ram_gb: int
    video_defaults: VideoDefaults
    models: list[TierModel]


class SettingsResponse(BaseModel):
    """Expose the hardware tier and transcription formatting."""

    tier: TierId | None = Field(
        description="The selected hardware tier; null until the user chooses one."
    )
    tiers: list[TierSetting]
    whisper_formatting: bool = Field(
        description="Normalize speech transcripts with S1-mini by Superwhisper; retain raw text."
    )


class SettingsRequest(BaseModel):
    """Persist the hardware tier and transcription formatting."""

    model_config = ConfigDict(extra="forbid")
    tier: TierId | None = None
    license_acknowledged: bool = Field(
        default=False,
        description="The user read the license notices of the tier's checkpoints.",
    )
    whisper_formatting: bool | None = None


@router.get("", operation_id="getSettings")
def get_settings() -> SettingsResponse:
    """Read the shared model store, including each tier's download progress.

    Raise HTTP 503 if the worker status is unavailable; this read does not load
    a model or configure another modality."""
    try:
        status = inference.models_status()
    except InferenceFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    checkpoints = {model.id: model for model in status.models}
    tiers = [
        TierSetting(
            id=tier.id,
            name=tier.name,
            vram_gb=tier.vram_gb,
            ram_gb=tier.ram_gb,
            video_defaults=tier.video_defaults,
            models=[
                TierModel(
                    modality=modality,
                    model=checkpoints.get(model_id) if model_id is not None else None,
                )
                for modality, model_id in tier.models
            ],
        )
        for tier in status.tiers
    ]
    return SettingsResponse(
        tier=status.selected_tier,
        tiers=tiers,
        whisper_formatting=status.whisper_formatting,
    )


@router.put(
    "",
    description="Persist the hardware tier and queue downloads of its checkpoints, and transcription formatting. The tier selects the language, image and video models and unloads a resident language model of another tier after earlier chats finish.",
    operation_id="updateSettings",
)
async def update_settings(body: SettingsRequest) -> SettingsResponse:
    """Persist settings and return the refreshed view.

    The worker validates and persists each change at the queue head; an
    omitted field leaves its setting unchanged. Choosing a tier queues its
    incomplete downloads in the same FIFO; poll this endpoint for progress.
    Requests load the tier's models on demand."""
    try:
        if body.tier is not None:
            await inference.select_tier(body.tier)
            await inference.download_tier(body.tier, body.license_acknowledged)
        if body.whisper_formatting is not None:
            await inference.submit(
                WhisperFormattingTask(enabled=body.whisper_formatting),
                feature="whisper_formatting",
                operation="configure",
            )
        return get_settings()
    except InferenceFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
