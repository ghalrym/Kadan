"""Kadan model loading controls, separate from versioned inference endpoints."""
from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from api.inference.errors import InferenceFailure
from api.services.chat_runtime import chat_runtime

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
    return ModelLifecycleStatus(**chat_runtime.status())


@router.post('/load', status_code=202, operation_id='loadSelectedModel')
async def load_selected_model(body: ModelLoadRequest | None = None) -> ModelLifecycleStatus:
    """Accept a saved-selection load, or atomically configure and load the supplied target.
    Validate before switching; 202/loading is acceptance, not completed construction.
    Poll status for readiness or errors. Identical explicit requests reuse the current load.
    """
    try:
        options = {} if body is None else {'model_id': body.model_id}
        if body is not None and 'context_limit' in body.model_fields_set:
            options['context_limit'] = body.context_limit
        return ModelLifecycleStatus(**await chat_runtime.load(**options))
    except InferenceFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


@router.post('/unload', operation_id='unloadSelectedModel')
async def unload_selected_model() -> ModelLifecycleStatus:
    """Request cooperative cancellation and wait for model cleanup before returning unloaded state."""
    return ModelLifecycleStatus(**await chat_runtime.unload())
