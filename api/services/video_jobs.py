"""Process-local video jobs with atomic output publication and cooperative cancellation."""
from datetime import datetime, timezone
import importlib
import os
from pathlib import Path
import threading
import uuid

from api.inference.resources import ResourceCancelled
from api.pydantic_models.media import VideoJob


class VideoJobs:
    def __init__(self, root=None, factory=None):
        self.root = Path(root or os.getenv('KADAN_VIDEO_DIR', '~/.local/share/kadan/videos')).expanduser()
        self.factory = factory or self._provider
        self._lock = threading.RLock()
        self._jobs = {}
        self._cancel = {}
        self._thread = None
        self._closed = False

    @staticmethod
    def _provider(model_id):
        """Resolve native providers without importing their worker dependencies into the API."""
        if model_id == 'wan22-ti2v-5b':
            module = importlib.import_module('api.inference.wan')
            return module.WanProvider(model_id, task='ti2v-5B')
        if model_id == 'ltx-2.5-distilled':
            try:
                module = importlib.import_module('api.inference.ltx')
            except ImportError as exc:
                raise RuntimeError('The LTX native provider is not installed.') from exc
            return module.LTXProvider()
        if model_id == 'h3-fl2va':
            try:
                module = importlib.import_module('api.inference.h3')
            except ImportError as exc:
                raise RuntimeError('The H3 native provider is not installed.') from exc
            return module.H3Provider(model_id)
        raise ValueError('Unknown native video model')

    def submit(self, model_id, spec):
        """Validate before queueing; allow one native video worker per API process."""
        provider = self.factory(model_id)
        provider.validate(spec)
        with self._lock:
            if self._closed:
                raise RuntimeError('Video service is shutting down.')
            if self._thread and self._thread.is_alive():
                raise ValueError('A video job is already running. Cancel it or wait for completion.')
            self.root.mkdir(parents=True, exist_ok=True)
            job_id = uuid.uuid4().hex
            job = VideoJob(id=job_id, prompt=spec.prompt, duration=f'{spec.duration}s',
                resolution=spec.resolution, aspect={'16:9': 'wide', '9:16': 'portrait', '1:1': 'square'}[spec.aspect],
                fps=str(spec.fps), progress=0, time=datetime.now(timezone.utc).isoformat(),
                status='Queued', thumbnail='', progressText='Queued')
            event = threading.Event()
            self._jobs[job_id] = job
            self._cancel[job_id] = event
            self._thread = threading.Thread(target=self._run, args=(job_id, provider, spec, event), daemon=True)
            try:
                self._thread.start()
            except BaseException:
                del self._jobs[job_id]
                del self._cancel[job_id]
                raise
            return job.model_copy(deep=True)

    def _update(self, job_id, **values):
        with self._lock:
            self._jobs[job_id] = self._jobs[job_id].model_copy(update=values)

    def _run(self, job_id, provider, spec, event):
        staging = self.root / f'.{job_id}.partial.mp4'
        destination = self.root / f'{job_id}.mp4'
        try:
            self._update(job_id, status='Rendering', progress_text='Rendering')
            provider.generate(spec, staging, event)
            with self._lock:
                if event.is_set():
                    raise ResourceCancelled('Video generation cancelled')
                if not staging.is_file() or staging.stat().st_size == 0:
                    raise RuntimeError('The provider returned no video.')
                staging.replace(destination)
                self._update(job_id, status='Done', progress=100, progress_text='Complete',
                             output_url=f'/v1/videos/{job_id}/content')
        except ResourceCancelled:
            self._update(job_id, status='Cancelled', progress_text='Cancelled')
        except Exception as exc:
            self._update(job_id, status='Failed', progress_text='Failed', error=str(exc))
        finally:
            staging.unlink(missing_ok=True)
            with self._lock:
                self._cancel.pop(job_id, None)

    def list(self):
        with self._lock:
            return [job.model_copy(deep=True) for job in reversed(list(self._jobs.values()))]

    def get(self, job_id):
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id].model_copy(deep=True)

    def cancel(self, job_id):
        with self._lock:
            job = self.get(job_id)
            event = self._cancel.get(job_id)
            if event:
                event.set()
            return job

    def content(self, job_id):
        job = self.get(job_id)
        if job.status != 'Done':
            raise KeyError(job_id)
        return self.root / f'{job.id}.mp4'

    def close(self):
        """Stop workers before allowing their owning API process to finish shutdown."""
        with self._lock:
            self._closed = True
            for event in self._cancel.values():
                event.set()
            thread = self._thread
        if thread:
            thread.join()


video_jobs = VideoJobs()
