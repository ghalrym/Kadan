"""Kadan model loading controls, separate from versioned inference endpoints."""
from typing import Literal
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from api.inference.errors import InferenceFailure
from api.services.model_downloads import model_manager
from api.services.inference import inference
from api.services.inference_http import infer

router = APIRouter(prefix='/model-lifecycle', tags=['Model lifecycle'])


class ModelLifecycleStatus(BaseModel):
    state: Literal['unloaded', 'loading', 'ready', 'offloaded', 'unloading', 'error']
    model_id: str | None = None
    error: str | None = None
    memory: dict | None = None
    configured_context_limit: int | None = None
    effective_context_limit: int | None = None
    supported_context_limit: int | None = None
    max_output_tokens: int = 256


class ModelLoadRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model_id: Literal['small', 'medium', 'large']
    context_limit: StrictInt | None = Field(default=None, ge=1, le=2**31 - 1,
        description='Omit to retain saved/default context; explicit null uses the architecture maximum.')


@router.get('', operation_id='getModelLifecycleStatus')
def get_model_lifecycle() -> ModelLifecycleStatus:
    """Return current model state, context limits and shared-memory accounting without loading a model."""
    return ModelLifecycleStatus(**inference.status())


@router.post('/load', status_code=202, operation_id='loadSelectedModel')
async def load_selected_model(request: Request, body: ModelLoadRequest | None = None) -> ModelLifecycleStatus:
    """Prepare and load through the same FIFO as inference, retaining cancellation ownership."""
    try:
        if body is None:
            selected = model_manager._read_selected_model_id()
            if selected is None:
                raise InferenceFailure('Select a language model before loading.', 422)
            body = ModelLoadRequest(model_id=selected)
        return ModelLifecycleStatus(**await infer(request, inference.submit(body, feature='llm', operation='load')))
    except InferenceFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


@router.post('/unload', operation_id='unloadSelectedModel')
async def unload_selected_model() -> ModelLifecycleStatus:
    """Request cooperative cancellation and wait for model cleanup before returning unloaded state."""
    try:
        return ModelLifecycleStatus(**await inference.unload())
    except InferenceFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
