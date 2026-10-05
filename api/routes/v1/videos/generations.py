from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from api.pydantic_models.media import VideoJob
from api.inference.video import VideoSpec
from api.services.video_jobs import video_jobs
from api.services.video_inputs import video_inputs

router = APIRouter(prefix="/v1/videos/generations", tags=["Videos"])

class VideoGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["ltx-2.5-distilled", "h3-fl2va", "wan22-i2v-a14b"] = "ltx-2.5-distilled"
    seed: int = Field(default=42, ge=0, le=4294967295)
    image_id: str | None = None
    audio_id: str | None = None
    video_id: str | None = None
    prompt: str = Field(min_length=1, max_length=8000, pattern=r"\S")
    negative_prompt: str = Field(default="", max_length=8000)
    duration: int = Field(default=8, gt=0, le=120)
    fps: int = Field(default=24, gt=0, le=120)
    resolution: Literal["480p", "720p", "768p", "1080p"] = "720p"
    aspect: Literal["16:9", "9:16", "1:1"] = "16:9"


class VideoGenerationResponse(BaseModel):
    job: VideoJob


@router.post("", status_code=202, operation_id="generateVideo",
             responses={503: {"description": "Video provider unavailable"}})
def generate_video(body: VideoGenerationRequest) -> VideoGenerationResponse:
    """Queue a validated native generation and return its actual job identifier."""
    try:
        if (body.image_id or body.audio_id or body.video_id) and not body.model.startswith('wan22-'):
            raise ValueError('This video provider currently accepts text conditioning only.')
        spec = VideoSpec(**body.model_dump(exclude={'model', 'image_id', 'audio_id', 'video_id'}),
            image_path=video_inputs.resolve(body.image_id, 'image'),
            audio_path=video_inputs.resolve(body.audio_id, 'audio'),
            video_path=video_inputs.resolve(body.video_id, 'video'))
        return VideoGenerationResponse(job=video_jobs.submit(body.model, spec))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=503, detail=f"{exc} No job was queued.") from exc
