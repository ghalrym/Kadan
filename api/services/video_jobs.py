"""Video presentation and artifact history; execution belongs to the native worker."""

from datetime import datetime, timezone
import os
from pathlib import Path
import threading

from api.pydantic_models.media import VideoJob


class VideoJobs:
    def __init__(self, root=None):
        self.root = Path(
            root or os.getenv("KADAN_VIDEO_DIR", "~/.local/share/kadan/videos")
        ).expanduser()
        self._lock = threading.RLock()
        self._jobs = {}

    def prepare(self, job_id, spec):
        with self._lock:
            if job_id not in self._jobs:
                self._jobs[job_id] = VideoJob(
                    id=job_id,
                    prompt=spec.prompt,
                    duration=f"{spec.duration}s",
                    resolution=spec.resolution,
                    aspect={"16:9": "wide", "9:16": "portrait", "1:1": "square"}[
                        spec.aspect
                    ],
                    fps=str(spec.fps),
                    progress=0,
                    time=datetime.now(timezone.utc).isoformat(),
                    status="Queued",
                    thumbnail="",
                    progressText="Queued",
                )
            return self.get(job_id)

    def update(self, job_id, **values):
        with self._lock:
            self._jobs[job_id] = self._jobs[job_id].model_copy(update=values)

    def list(self):
        with self._lock:
            return [
                job.model_copy(deep=True) for job in reversed(list(self._jobs.values()))
            ]

    def get(self, job_id):
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id].model_copy(deep=True)

    def content(self, job_id):
        job = self.get(job_id)
        if job.status != "Done":
            raise KeyError(job_id)
        return self.root / f"{job.id}.mp4"


video_jobs = VideoJobs()
