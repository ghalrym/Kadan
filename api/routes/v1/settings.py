"""Settings view of the worker-owned model store."""

from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from api.pydantic_models.settings import ModelSetting
from api.services.inference import inference
from api.services.inference_errors import InferenceFailure
from api.pydantic_models.inference_task import ModelSelectionTask

router = APIRouter(prefix="/v1/settings", tags=["Settings"])


class SettingsResponse(BaseModel):
    """Expose real catalog selection and null for unavailable transcription settings."""

    models: list[ModelSetting]
    whisper_formatting: bool | None = Field(
        default=None, description="Null: no transcription provider is configured."
    )


class SettingsRequest(BaseModel):
    """Accept only the supported LLM selection; reject unknown top-level fields."""

    model_config = ConfigDict(extra="forbid")
    models: dict[Literal["LLM"], Literal["small", "medium", "large"]]
    whisper_formatting: bool | None = None


@router.get("", operation_id="getSettings")
def get_settings() -> SettingsResponse:
    """Read the shared model store and return the catalog selection.

    Raise HTTP 503 if the stored selection is outside the catalog; this read does
    not load a model or configure another modality."""
    try:
        status = inference.models_status()
    except InferenceFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    llm_options = [model.id for model in status.models if model.kind == "llm"]
    return SettingsResponse(
        models=[
            ModelSetting(
                label="Language model",
                type="LLM",
                selected=status.selected_model_id,
                options=llm_options,
            )
        ]
    )


@router.put(
    "",
    description="Persist LLM selection through the same store as /v1/models/selection. Selection requires a completed download. Other modalities are not configured.",
    operation_id="updateSettings",
)
async def update_settings(body: SettingsRequest) -> SettingsResponse:
    """Persist a completed LLM selection and return the refreshed settings.

    The worker validates and persists selection at the queue head. An empty models
    mapping leaves selection unchanged. Selection alone does not load inference."""
    if body.whisper_formatting is not None:
        raise HTTPException(
            503,
            "Transcription settings are unavailable: no transcription provider is configured.",
        )
    try:
        if "LLM" in body.models:
            await inference.submit(
                ModelSelectionTask(model_id=body.models["LLM"]),
                feature="model_selection",
                operation="configure",
            )
        return get_settings()
    except InferenceFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
