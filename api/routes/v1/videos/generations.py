from typing import Literal
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field
from api.pydantic_models.media import VideoJob

router = APIRouter(prefix="/v1/videos/generations", tags=["Videos"])

MOCK_VIDEO_JOB = VideoJob(
    id='vid_77c0e4',
    prompt=('Macro shot of coffee being poured into a glass cup, steam rising, morning '
            'light'),
    duration='4s',
    resolution='1080p',
    aspect='portrait',
    fps='30',
    progress=0,
    time='14:27',
    status='Queued',
    thumbnail='In queue',
    progressText='waiting',
)


class VideoGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    duration: int = Field(default=8, gt=0)
    fps: int = Field(default=24, gt=0)
    resolution: Literal["480p", "720p", "1080p"] = "720p"
    aspect: Literal["16:9", "9:16", "1:1"] = "16:9"


class VideoGenerationResponse(BaseModel):
    job: VideoJob


@router.post("", status_code=202)
def generate_video(body: VideoGenerationRequest) -> VideoGenerationResponse:
    return VideoGenerationResponse(job=MOCK_VIDEO_JOB)
