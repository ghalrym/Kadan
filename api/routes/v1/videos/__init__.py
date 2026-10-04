"""Video job queries. No video provider or persisted jobs are available yet."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.pydantic_models.media import VideoJob

router = APIRouter(prefix="/v1/videos", tags=["Videos"])


class VideosResponse(BaseModel):
    jobs: list[VideoJob]


class VideoResponse(BaseModel):
    job: VideoJob


@router.get("", operation_id="listVideos")
def list_videos() -> VideosResponse:
    """Return an empty queue because no video jobs are persisted or scheduled."""
    return VideosResponse(jobs=[])


@router.get("/{video_id}", operation_id="getVideo", responses={404: {"description": "Video not found"}})
def get_video(video_id: str) -> VideoResponse:
    """Report HTTP 404 for the requested ID; no provider-backed video jobs exist."""
    raise HTTPException(status_code=404, detail="Video not found")
