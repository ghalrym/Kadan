"""Compatibility settings view backed by the real model selection store."""
from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from api.pydantic_models.settings import ModelSetting
from api.services.model_catalog import CATALOG
from api.services.model_downloads import BusyError, model_manager

router = APIRouter(prefix="/v1/settings", tags=["Settings"])


class SettingsResponse(BaseModel):
    models: list[ModelSetting]
    whisper_formatting: bool | None = Field(default=None, description="Null: no transcription provider is configured.")


class SettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models: dict[Literal['LLM'], Literal['small', 'medium', 'large']]
    whisper_formatting: bool | None = None


@router.get("", operation_id="getSettings")
def get_settings() -> SettingsResponse:
    status = model_manager.status()
    if status['selected_model_id'] is not None and status['selected_model_id'] not in CATALOG:
        raise HTTPException(503, 'Stored model selection is invalid. Select a catalog model in Settings.')
    return SettingsResponse(models=[ModelSetting(
        label='Language model', type='LLM', selected=status['selected_model_id'],
        options=list(CATALOG),
    )])


@router.put("", description="Persist LLM selection through the same store as /v1/models/selection. Selection requires a completed download. Other modalities are not configured.", operation_id="updateSettings")
def update_settings(body: SettingsRequest) -> SettingsResponse:
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
