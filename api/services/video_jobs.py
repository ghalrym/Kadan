"""Process-local video jobs with atomic output publication and cooperative cancellation."""
from datetime import datetime, timezone
import importlib
import os
from pathlib import Path
import threading

from api.inference.resources import ResourceCancelled
from api.pydantic_models.media import VideoJob
from api.services.runtime import RuntimeFailure


class VideoJobs:
    def __init__(self, root=None, factory=None):
        self.root = Path(root or os.getenv('KADAN_VIDEO_DIR', '~/.local/share/kadan/videos')).expanduser()
        self.factory = factory or self._provider
        self._lock = threading.RLock()
        self._jobs = {}
        self._cancel = {}
        self._providers = {}
        self._closed = False

    @staticmethod
    def _provider(model_id):
        """Resolve native providers without importing their worker dependencies into the API."""
        if model_id == 'ltx-2.5-distilled':
            try:
                module = importlib.import_module('api.inference.video.ltx')
            except ImportError as exc:
                raise RuntimeError('The LTX native provider is not installed.') from exc
            return module.LTXProvider()
        if model_id == 'h3-fl2va-int8-turbo':
            try:
                module = importlib.import_module('api.inference.video.h3')
            except ImportError as exc:
                raise RuntimeError('The H3 native provider is not installed.') from exc
            return module.H3Provider(model_id)
        raise ValueError('Unknown native video model')

    def provider(self, model_id):
        if model_id not in self._providers:
            self._providers[model_id] = self.factory(model_id)
        return self._providers[model_id]

    def load(self, model_id, cancel):
        self.provider(model_id).load(cancel)

    def offload_to_ram(self, cancel=None):
        for provider in self._providers.values():
            provider.offload_to_ram(cancel)

    def unload(self):
        """Release model adapters while preserving job metadata and output history."""
        with self._lock:
            for provider in self._providers.values():
                close = getattr(provider, 'close', None)
                if close is not None:
                    close()
            self._providers.clear()

    def validate(self, model_id, spec):
        if model_id not in self._providers:
            self._providers[model_id] = self.factory(model_id)
        self._providers[model_id].validate(spec)

    def prepare(self, job_id, spec):
        with self._lock:
            if job_id not in self._jobs:
                self._jobs[job_id] = VideoJob(id=job_id, prompt=spec.prompt, duration=f'{spec.duration}s',
                    resolution=spec.resolution, aspect={'16:9': 'wide', '9:16': 'portrait', '1:1': 'square'}[spec.aspect],
                    fps=str(spec.fps), progress=0, time=datetime.now(timezone.utc).isoformat(),
                    status='Queued', thumbnail='', progressText='Queued')
            return self.get(job_id)

    def run(self, job_id, model_id, spec, event):
        """Run only inside the shared API consumer; no independent video worker."""
        self.validate(model_id, spec)
        self.prepare(job_id, spec)
        with self._lock:
            if self._closed or self._jobs[job_id].status == 'Cancelled':
                raise ResourceCancelled('Video generation cancelled')
            self.root.mkdir(parents=True, exist_ok=True)
            self._cancel[job_id] = event
        self._run(job_id, self._providers[model_id], spec, event)
        job = self.get(job_id)
        if job.status == 'Failed':
            raise RuntimeFailure(job.error, 502)
        if job.status == 'Cancelled':
            raise ResourceCancelled('Video generation cancelled')
        return job.model_dump(mode='json')

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
            elif job.status == 'Queued':
                self._update(job_id, status='Cancelled', progress_text='Cancelled')
            return job

    def content(self, job_id):
        job = self.get(job_id)
        if job.status != 'Done':
            raise KeyError(job_id)
        return self.root / f'{job.id}.mp4'

    def close(self):
        """The shared consumer closes before this service during API shutdown."""
        with self._lock:
            self._closed = True
            for event in self._cancel.values():
                event.set()
            for provider in self._providers.values():
                close = getattr(provider, 'close', None)
                if close is not None:
                    close()


video_jobs = VideoJobs()
