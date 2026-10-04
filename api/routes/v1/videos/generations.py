from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from api.pydantic_models.media import VideoJob

router = APIRouter(prefix="/v1/videos/generations", tags=["Videos"])

class VideoGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=8000, pattern=r"\S")
    negative_prompt: str = Field(default="", max_length=8000)
    duration: int = Field(default=8, gt=0, le=120)
    fps: int = Field(default=24, gt=0, le=120)
    resolution: Literal["480p", "720p", "1080p"] = "720p"
    aspect: Literal["16:9", "9:16", "1:1"] = "16:9"


class VideoGenerationResponse(BaseModel):
    job: VideoJob


@router.post("", status_code=202, operation_id="generateVideo",
             responses={503: {"description": "Video provider unavailable"}})
def generate_video(body: VideoGenerationRequest) -> VideoGenerationResponse:
    raise HTTPException(status_code=503, detail="Video generation provider is not configured. No job was queued.")
