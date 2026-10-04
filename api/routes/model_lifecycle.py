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


@router.get('', operation_id='getModelLifecycleStatus')
def get_model_lifecycle() -> ModelLifecycleStatus:
    return ModelLifecycleStatus(**runtime_manager.status())


@router.post('/load', status_code=202, operation_id='loadSelectedModel')
async def load_selected_model() -> ModelLifecycleStatus:
    try:
        return ModelLifecycleStatus(**await runtime_manager.load())
    except RuntimeFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


@router.post('/unload', operation_id='unloadSelectedModel')
async def unload_selected_model() -> ModelLifecycleStatus:
    return ModelLifecycleStatus(**await runtime_manager.unload())
