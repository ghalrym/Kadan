"""Kadan model loading controls, separate from versioned inference endpoints."""
from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.services.runtime import RuntimeFailure, runtime_manager

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


@router.get('', operation_id='getModelLifecycleStatus')
def get_model_lifecycle() -> ModelLifecycleStatus:
    """Return current model state, context limits and shared-memory accounting without loading a model."""
    return ModelLifecycleStatus(**runtime_manager.status())


@router.post('/load', status_code=202, operation_id='loadSelectedModel')
async def load_selected_model() -> ModelLifecycleStatus:
    """Accept loading of the selected complete checkpoint and return its initial state; map
    lifecycle conflicts and validation failures to HTTP errors.
    """
    try:
        return ModelLifecycleStatus(**await runtime_manager.load())
    except RuntimeFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


@router.post('/unload', operation_id='unloadSelectedModel')
async def unload_selected_model() -> ModelLifecycleStatus:
    """Request cooperative cancellation and wait for model cleanup before returning unloaded state."""
    return ModelLifecycleStatus(**await runtime_manager.unload())
