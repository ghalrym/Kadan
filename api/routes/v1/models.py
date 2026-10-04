from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from api.services.model_downloads import BusyError, model_manager

router = APIRouter(prefix='/v1/models', tags=['models'])


class ModelStatus(BaseModel):
    id: str
    repo_id: str
    revision: str
    license: str
    estimated_bytes: int
    status: Literal['not_downloaded', 'downloading', 'cancelling', 'cancelled', 'failed', 'complete']
    downloaded_bytes: int
    total_bytes: int
    error: str | None
    context_limit: int | None
    architecture_context_limit: int | None


class ModelsResponse(BaseModel):
    models: list[ModelStatus]
    selected_model_id: str | None


class SelectionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model_id: Literal['small', 'medium', 'large']


class ContextRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    context_limit: StrictInt | None = Field(ge=1, le=2**31 - 1,
        description='Total context tokens; null uses the checkpoint architecture maximum on load.')


@router.put('/{model_id}/context', response_model=ModelsResponse)
def configure_context(model_id: str, body: ContextRequest):
    try:
        model_manager.set_context(model_id, body.context_limit)
        return model_manager.status()
    except BusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(503, 'Model storage is unavailable; check directory permissions and disk space') from exc


@router.get('', response_model=ModelsResponse)
def list_models():
    return model_manager.status()


@router.put('/selection', response_model=ModelsResponse)
def select_model(body: SelectionRequest):
    try:
        model_manager.select(body.model_id)
        return model_manager.status()
    except BusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(503, 'Model storage is unavailable; check directory permissions and disk space') from exc


@router.post('/{model_id}/download', response_model=ModelsResponse, status_code=202)
def download_model(model_id: str):
    try:
        model_manager.start(model_id)
        return model_manager.status()
    except BusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(503, 'Model storage is unavailable; check directory permissions and disk space') from exc


@router.delete('/{model_id}/download', response_model=ModelsResponse, status_code=202)
def cancel_download(model_id: str):
    try:
        model_manager.cancel(model_id)
        return model_manager.status()
    except BusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(503, 'Model storage is unavailable; check directory permissions and disk space') from exc
