"""Hikari model loading controls, separate from versioned inference endpoints."""

from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.services.inference_errors import InferenceFailure
from api.services.inference import inference

router = APIRouter(prefix="/model-lifecycle", tags=["Model lifecycle"])


class ModelLifecycleStatus(BaseModel):
    state: Literal["unloaded", "loading", "ready", "offloaded", "unloading", "error"]
    model_id: str | None = None
    error: str | None = None
    memory: dict | None = None
    configured_context_limit: int | None = None
    effective_context_limit: int | None = None
    supported_context_limit: int | None = None
    max_output_tokens: int | None = None


@router.get("", operation_id="getModelLifecycleStatus")
def get_model_lifecycle() -> ModelLifecycleStatus:
    """Return current model state, context limits and shared-memory accounting without loading a model."""
    return ModelLifecycleStatus(**inference.language_status())


@router.post("/unload", operation_id="unloadSelectedModel")
async def unload_selected_model() -> ModelLifecycleStatus:
    """Request cooperative cancellation and wait for model cleanup before returning unloaded state."""
    try:
        return ModelLifecycleStatus(**await inference.unload())
    except InferenceFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
