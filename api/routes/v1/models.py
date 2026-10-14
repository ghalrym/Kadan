from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictBool
from api.pydantic_models.model_store import CheckpointStatus
from api.pydantic_models.inference_task import LlmModel, ModelContextTask

from api.services.inference import inference
from api.services.inference_errors import InferenceFailure

router = APIRouter(prefix="/v1/models", tags=["models"])


class ModelsResponse(BaseModel):
    models: list[CheckpointStatus]
    selected_model_id: str | None
    selected_image: str | None


class SelectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: LlmModel


class DownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    license_acknowledged: StrictBool = False


class ContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    context_limit: StrictInt | None = Field(
        ge=1,
        le=2**31 - 1,
        description="Total context tokens; null uses the checkpoint architecture maximum on load.",
    )


@router.put("/{model_id}/context", response_model=ModelsResponse)
async def configure_context(model_id: str, body: ContextRequest) -> ModelsResponse:
    try:
        await inference.submit(
            ModelContextTask(model_id=model_id, context_limit=body.context_limit),
            feature="model_context",
            operation="configure",
        )
        return list_models()
    except InferenceFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


@router.get("", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    """Return the worker's latest model snapshot and observed download progress."""
    try:
        snapshot = inference.models_status()
        return ModelsResponse(
            models=snapshot.models,
            selected_model_id=snapshot.selected_model_id,
            selected_image=snapshot.selected_image,
        )
    except InferenceFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


@router.put("/selection", response_model=ModelsResponse)
async def select_model(body: SelectionRequest) -> ModelsResponse:
    try:
        await inference.submit(body, feature="model_selection", operation="configure")
        return list_models()
    except InferenceFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


@router.post("/{model_id}/download", response_model=ModelsResponse, status_code=202)
async def download_model(
    model_id: str, body: DownloadRequest | None = None
) -> ModelsResponse:
    """Queue a worker download; poll model status for progress and completion."""
    try:
        await inference.download(model_id, bool(body and body.license_acknowledged))
        return list_models()
    except InferenceFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


@router.delete("/{model_id}/download", response_model=ModelsResponse, status_code=202)
async def cancel_download(model_id: str) -> ModelsResponse:
    try:
        await inference.cancel_download(model_id)
        return list_models()
    except InferenceFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
