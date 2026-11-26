"""Native video job queries, cancellation, and completed media."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import FileResponse
from api.services.video_jobs import video_jobs
from api.services.inference import inference
from api.services.inference_errors import InferenceFailure
from api.pydantic_models.media import VideoJob

router = APIRouter(prefix="/v1/videos", tags=["Videos"])


class VideosResponse(BaseModel):
    jobs: list[VideoJob]


class VideoResponse(BaseModel):
    job: VideoJob


def public_job(job: VideoJob, request: Request) -> VideoJob:
    """Prefix the stored content path with where this API is mounted, e.g. /hikari."""
    root = request.scope.get("root_path", "")
    if root and job.output_url and job.output_url.startswith("/v1/"):
        return job.model_copy(update={"output_url": root + job.output_url})
    return job


@router.get("", operation_id="listVideos")
async def list_videos(request: Request) -> VideosResponse:
    """Return jobs submitted to this API process."""
    try:
        return VideosResponse(
            jobs=[
                public_job(video_jobs.get(job.id), request) for job in video_jobs.list()
            ]
        )
    except InferenceFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


@router.get(
    "/{video_id}",
    operation_id="getVideo",
    responses={404: {"description": "Video not found"}},
)
async def get_video(video_id: str, request: Request) -> VideoResponse:
    """Return the real worker state, including errors and cancellation."""
    try:
        return VideoResponse(job=public_job(video_jobs.get(video_id), request))
    except InferenceFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Video not found") from exc


@router.delete("/{video_id}", operation_id="cancelVideo")
async def cancel_video(video_id: str, request: Request) -> VideoResponse:
    try:
        await inference.cancel(video_id)
        return VideoResponse(job=public_job(video_jobs.get(video_id), request))
    except InferenceFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Video not found") from exc


@router.get("/{video_id}/content", operation_id="getVideoContent")
def get_video_content(video_id: str):
    try:
        path = video_jobs.content(video_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail="Completed video not found"
        ) from exc
    return FileResponse(path, media_type="video/mp4", filename=f"{video_id}.mp4")
