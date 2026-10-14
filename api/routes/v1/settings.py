"""Settings view of the worker-owned model store."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from api.pydantic_models.settings import ModelSetting
from api.services.inference import inference
from api.services.inference_errors import InferenceFailure
from api.pydantic_models.inference_task import (
    ImageModel,
    ImageSelectionTask,
    LlmModel,
)

router = APIRouter(prefix="/v1/settings", tags=["Settings"])


class SettingsResponse(BaseModel):
    """Expose real catalog selection and null for unavailable transcription settings."""

    models: list[ModelSetting]
    whisper_formatting: bool | None = Field(
        default=None, description="Null: no transcription provider is configured."
    )


class SettingsModels(BaseModel):
    """Per-modality selections; an omitted modality is left unchanged."""

    model_config = ConfigDict(extra="forbid")
    LLM: LlmModel | None = None
    Image: ImageModel | None = None


class SettingsRequest(BaseModel):
    """Accept only the supported LLM and image selections; reject unknown fields."""

    model_config = ConfigDict(extra="forbid")
    models: SettingsModels
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
    image_options = [model.id for model in status.models if model.kind == "image"]
    return SettingsResponse(
        models=[
            ModelSetting(
                label="Language model",
                type="LLM",
                selected=status.selected_model_id,
                options=llm_options,
            ),
            ModelSetting(
                label="Image generation",
                type="Image",
                selected=status.selected_image,
                options=image_options,
            ),
        ]
    )


@router.put(
    "",
    description="Persist LLM and image selections in the worker model store. Selection requires a completed download and unloads a language model resident for a different checkpoint. Other modalities are not configured.",
    operation_id="updateSettings",
)
async def update_settings(body: SettingsRequest) -> SettingsResponse:
    """Persist completed LLM and image selections and return the refreshed settings.

    The worker validates and persists selection at the queue head. An omitted
    modality leaves its selection unchanged. Selection alone does not load
    inference; requests load the selected model on demand. A language model
    resident for a different checkpoint is unloaded after earlier chats finish,
    so status never reports it. Context changes never unload."""
    if body.whisper_formatting is not None:
        raise HTTPException(
            503,
            "Transcription settings are unavailable: no transcription provider is configured.",
        )
    try:
        if body.models.LLM is not None:
            await inference.select_model(body.models.LLM)
        if body.models.Image is not None:
            await inference.submit(
                ImageSelectionTask(model_id=body.models.Image),
                feature="image_selection",
                operation="configure",
            )
        return get_settings()
    except InferenceFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
