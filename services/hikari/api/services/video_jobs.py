"""Video presentation and artifact history; execution belongs to the native worker.

The worker returns each finished video inline, so its file is written here.
"""

import os
import threading
from datetime import UTC, datetime
from pathlib import Path

from api.pydantic_models.media import VideoJob


class VideoJobs:
    def __init__(self, root=None):
        self.root = Path(
            root
            or os.getenv(
                "HIKARI_VIDEO_DIR",
                os.getenv("KADAN_VIDEO_DIR", "~/.local/share/kadan/videos"),
            )
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
                    time=datetime.now(UTC).isoformat(),
                    status="Queued",
                    thumbnail="",
                    progressText="Queued",
                )
            return self.get(job_id)

    def update(self, job_id, **values):
        with self._lock:
            self._jobs[job_id] = self._jobs[job_id].model_copy(update=values)

    def finish_error(self, job_id, *, cancelled: bool, error: str | None = None):
        """Settle unsuccessful delivery without replacing a confirmed outcome."""
        with self._lock:
            job = self._jobs[job_id]
            if job.status in ("Done", "Failed", "Cancelled"):
                return
            status = "Cancelled" if cancelled else "Failed"
            self._jobs[job_id] = job.model_copy(
                update={"status": status, "progress_text": status, "error": error}
            )

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

    def store(self, job_id, data: bytes):
        """Write a finished video atomically where content() serves it."""
        self.root.mkdir(parents=True, exist_ok=True)
        staging = self.root / f".{job_id}.partial.mp4"
        try:
            staging.write_bytes(data)
            staging.replace(self.root / f"{job_id}.mp4")
        finally:
            staging.unlink(missing_ok=True)

    def content(self, job_id):
        job = self.get(job_id)
        if job.status != "Done":
            raise KeyError(job_id)
        return self.root / f"{job.id}.mp4"


video_jobs = VideoJobs()
