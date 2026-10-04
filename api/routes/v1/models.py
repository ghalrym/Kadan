from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from api.services.model_downloads import BusyError, DownloadStatus, ModelsStatus, model_manager

router = APIRouter(prefix='/v1/models', tags=['models'])


class ModelStatus(BaseModel):
    id: str
    repo_id: str
    revision: str
    license: str
    estimated_bytes: int
    status: DownloadStatus
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
def configure_context(model_id: str, body: ContextRequest) -> ModelsStatus:
    """Persist the model's context limit and return refreshed catalog status.

    Null uses the architecture maximum on load. Returns 409 while a runtime lease
    is held, 400 for invalid limits/model IDs, or 503 for unavailable storage."""
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
def list_models() -> ModelsStatus:
    """Return current catalog, download progress, saved selection, and context limits.

    Completeness is checked from local files; this does not start downloads or
    load models. Context/configuration errors propagate; an unreadable or
    invalid saved selection is represented as no selection."""
    return model_manager.status()


@router.put('/selection', response_model=ModelsResponse)
def select_model(body: SelectionRequest) -> ModelsStatus:
    """Save a completed model as the selection for a subsequent runtime load.

    Returns refreshed status, or 409 for a runtime lease, 400 for an incomplete
    checkpoint, and 503 for unavailable storage. Selection alone loads no tensors."""
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
def download_model(model_id: str) -> ModelsStatus:
    """Start a catalog checkpoint download and return status with HTTP 202.

    Acceptance is not completion; poll the catalog for progress or failure.
    Unknown models return 400; active/completed conflicts return 409 and storage
    failures return 503. Only pinned catalog checkpoints may be downloaded."""
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
def cancel_download(model_id: str) -> ModelsStatus:
    """Request cancellation and return status with HTTP 202 before cleanup finishes.

    Unknown models return 400, inactive jobs return 409, and storage errors return
    503. Completed checkpoints are never deleted by this endpoint."""
    try:
        model_manager.cancel(model_id)
        return model_manager.status()
    except BusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(503, 'Model storage is unavailable; check directory permissions and disk space') from exc
