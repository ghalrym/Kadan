from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from api.pydantic_models.media import VideoJob
from api.services.inference import inference
from api.inference.errors import InferenceFailure

router = APIRouter(prefix="/v1/videos/generations", tags=["Videos"])

class VideoGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["h3-fl2va-int8-turbo"] = "h3-fl2va-int8-turbo"
    seed: int = Field(default=42, ge=0, le=4294967295)
    prompt: str = Field(min_length=1, max_length=8000, pattern=r"\S")
    negative_prompt: Literal[""] = ""
    duration: int = Field(default=8, ge=4, le=15)
    fps: Literal[24] = 24
    resolution: Literal["480p", "768p"] = "768p"
    aspect: Literal["16:9", "9:16", "1:1"] = "16:9"


class VideoGenerationResponse(BaseModel):
    job: VideoJob


@router.post("", status_code=202, operation_id="generateVideo",
             responses={503: {"description": "Video provider unavailable"}})
async def generate_video(body: VideoGenerationRequest) -> VideoGenerationResponse:
    """Queue a validated native generation and return its actual job identifier."""
    try:
        return VideoGenerationResponse(job=await inference.submit(body, feature='video'))
    except InferenceFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=503, detail=f"{exc} No job was queued.") from exc
