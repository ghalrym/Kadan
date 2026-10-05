"""Compatibility settings view backed by the real model selection store."""
from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from api.pydantic_models.settings import ModelSetting
from api.services.model_catalog import CATALOG
from api.services.model_downloads import BusyError, model_manager

router = APIRouter(prefix="/v1/settings", tags=["Settings"])


class SettingsResponse(BaseModel):
    """Expose real catalog selection and null for unavailable transcription settings."""
    models: list[ModelSetting]
    whisper_formatting: bool | None = Field(default=None, description="Null: no transcription provider is configured.")


class SettingsRequest(BaseModel):
    """Accept only the supported LLM selection; reject unknown top-level fields."""
    model_config = ConfigDict(extra="forbid")
    models: dict[Literal['LLM'], Literal['small', 'medium', 'large']]
    whisper_formatting: bool | None = None


@router.get("", operation_id="getSettings")
def get_settings() -> SettingsResponse:
    """Read the shared model store and return the catalog selection.

    Raise HTTP 503 if the stored selection is outside the catalog; this read does
    not load a model or configure another modality."""
    status = model_manager.status()
    llm_options = [model_id for model_id, entry in CATALOG.items() if entry.kind == 'llm']
    if status['selected_model_id'] is not None and status['selected_model_id'] not in llm_options:
        raise HTTPException(503, 'Stored model selection is invalid. Select a catalog model in Settings.')
    return SettingsResponse(models=[ModelSetting(
        label='Language model', type='LLM', selected=status['selected_model_id'],
        options=llm_options,
    )])


@router.put("", description="Persist LLM selection through the same store as /v1/models/selection. Selection requires a completed download. Other modalities are not configured.", operation_id="updateSettings")
def update_settings(body: SettingsRequest) -> SettingsResponse:
    """Persist a completed LLM selection and return the refreshed settings.

    An active model lease yields 409, invalid selection yields 400, and storage
    failures or unsupported transcription settings yield 503. An empty models
    mapping leaves selection unchanged. Selection alone does not load inference."""
    if body.whisper_formatting is not None:
        raise HTTPException(503, 'Transcription settings are unavailable: no transcription provider is configured.')
    try:
        if 'LLM' in body.models:
            model_manager.select(body.models['LLM'])
        return get_settings()
    except BusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(503, 'Model settings storage is unavailable.') from exc
