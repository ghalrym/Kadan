from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.services.runtime import RuntimeFailure, runtime_manager

router = APIRouter(prefix='/v1/runtime', tags=['Runtime'])


class RuntimeStatus(BaseModel):
    state: Literal['unloaded', 'loading', 'ready', 'offloaded', 'unloading', 'error']
    model_id: str | None = None
    error: str | None = None
    memory: dict | None = None


@router.get('', operation_id='getRuntimeStatus')
def get_runtime_status() -> RuntimeStatus:
    return RuntimeStatus(**runtime_manager.status())


@router.post('/load', status_code=202, operation_id='loadRuntimeModel')
async def load_runtime_model() -> RuntimeStatus:
    try:
        return RuntimeStatus(**await runtime_manager.load())
    except RuntimeFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


@router.post('/unload', operation_id='unloadRuntimeModel')
async def unload_runtime_model() -> RuntimeStatus:
    return RuntimeStatus(**await runtime_manager.unload())
