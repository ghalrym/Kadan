"""Native video job queries, cancellation, and completed media."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from api.services.video_jobs import video_jobs
from api.memory_manager import memory_manager
from api.services.runtime import RuntimeFailure
from api.pydantic_models.media import VideoJob

router = APIRouter(prefix="/v1/videos", tags=["Videos"])


class VideosResponse(BaseModel):
    jobs: list[VideoJob]


class VideoResponse(BaseModel):
    job: VideoJob


@router.get("", operation_id="listVideos")
async def list_videos() -> VideosResponse:
    """Return jobs submitted to this API process."""
    try:
        return VideosResponse(jobs=[await memory_manager.video_job(job.id) for job in video_jobs.list()])
    except RuntimeFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


@router.get("/{video_id}", operation_id="getVideo", responses={404: {"description": "Video not found"}})
async def get_video(video_id: str) -> VideoResponse:
    """Return the real worker state, including errors and cancellation."""
    try:
        return VideoResponse(job=await memory_manager.video_job(video_id))
    except RuntimeFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Video not found") from exc


@router.delete("/{video_id}", operation_id="cancelVideo")
async def cancel_video(video_id: str) -> VideoResponse:
    try:
        await memory_manager.queue.cancel(video_id)
        return VideoResponse(job=video_jobs.cancel(video_id))
    except RuntimeFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Video not found") from exc


@router.get("/{video_id}/content", operation_id="getVideoContent")
def get_video_content(video_id: str):
    try:
        path = video_jobs.content(video_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Completed video not found") from exc
    return FileResponse(path, media_type="video/mp4", filename=f"{video_id}.mp4")
