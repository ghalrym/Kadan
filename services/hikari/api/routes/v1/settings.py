"""Fixed model assignments and worker-observed checkpoint downloads."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from api.pydantic_models.inference_task import (
    LlmModel,
    LlmSelectionTask,
    WhisperFormattingTask,
)
from api.pydantic_models.model_store import (
    CheckpointStatus,
    ModelAssignments,
)
from api.services.inference import inference
from api.services.inference_errors import InferenceFailure

router = APIRouter(prefix="/v1/settings", tags=["Settings"])


class SettingsResponse(BaseModel):
    models: list[CheckpointStatus]
    defaults: ModelAssignments
    whisper_formatting: bool


class SettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    whisper_formatting: bool | None = None
    llm_model: LlmModel | None = None


@router.get("", operation_id="getSettings")
def get_settings() -> SettingsResponse:
    try:
        status = inference.models_status()
    except InferenceFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    return SettingsResponse(
        models=status.models,
        defaults=status.defaults,
        whisper_formatting=status.whisper_formatting,
    )


@router.put("", operation_id="updateSettings")
async def update_settings(body: SettingsRequest) -> SettingsResponse:
    try:
        if body.whisper_formatting is not None:
            await inference.submit(
                WhisperFormattingTask(enabled=body.whisper_formatting),
                feature="whisper_formatting",
                operation="configure",
            )
        if body.llm_model is not None:
            await inference.submit(
                LlmSelectionTask(model_id=body.llm_model),
                feature="llm_selection",
                operation="configure",
            )
        return get_settings()
    except InferenceFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
