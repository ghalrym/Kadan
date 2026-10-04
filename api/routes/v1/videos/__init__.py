from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.pydantic_models.media import VideoJob

router = APIRouter(prefix="/v1/videos", tags=["Videos"])

MOCK_VIDEO_JOBS = [
    VideoJob(
        id='vid_3f9a21',
        prompt=('Slow dolly across a rain-soaked neon street at night, reflections on '
                'the asphalt, cinematic'),
        duration='8s',
        resolution='720p',
        aspect='wide',
        fps='24',
        progress=38,
        time='14:26',
        status='Rendering',
        thumbnail='Frame 72 / 192',
        progressText='38% · ~0:14 left',
    ),
    VideoJob(
        id='vid_77c0e4',
        prompt=('Macro shot of coffee being poured into a glass cup, steam rising, '
                'morning light'),
        duration='4s',
        resolution='1080p',
        aspect='portrait',
        fps='30',
        progress=0,
        time='14:27',
        status='Queued',
        thumbnail='In queue',
        progressText='waiting',
    ),
    VideoJob(
        id='vid_1b52d8',
        prompt='Drone shot rising over a foggy pine forest at sunrise',
        duration='12s',
        resolution='720p',
        aspect='wide',
        fps='24',
        progress=100,
        time='13:52',
        status='Done',
        thumbnail='12s · 720p',
        progressText='12s rendered',
    ),
]


class VideosResponse(BaseModel):
    jobs: list[VideoJob]


class VideoResponse(BaseModel):
    job: VideoJob


@router.get("", operation_id="listVideos")
def list_videos() -> VideosResponse:
    return VideosResponse(jobs=MOCK_VIDEO_JOBS)


@router.get("/{video_id}", operation_id="getVideo")
def get_video(video_id: str) -> VideoResponse:
    for job in MOCK_VIDEO_JOBS:
        if job.id == video_id:
            return VideoResponse(job=job)
    raise HTTPException(status_code=404, detail="Video not found")
