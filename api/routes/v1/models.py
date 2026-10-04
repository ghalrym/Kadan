from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
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


class ModelsResponse(BaseModel):
    models: list[ModelStatus]
    selected_model_id: str | None


class SelectionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model_id: Literal['small', 'medium', 'large']


def _perform(action, *args):
    try:
        action(*args)
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
    return _perform(model_manager.select, body.model_id)


@router.post('/{model_id}/download', response_model=ModelsResponse, status_code=202)
def download_model(model_id: str):
    return _perform(model_manager.start, model_id)


@router.delete('/{model_id}/download', response_model=ModelsResponse, status_code=202)
def cancel_download(model_id: str):
    return _perform(model_manager.cancel, model_id)
