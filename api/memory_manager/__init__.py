"""One Redis consumer coordinating six concrete native feature wrappers."""
from contextlib import AsyncExitStack
import logging
import os

from pydantic import ValidationError

from api.inference.llm.feature import LLMFeature
from api.inference.video.feature import VideoFeature
from api.inference.image.feature import ImageFeature
from api.inference.stt.feature import STTFeature
from api.inference.tts.feature import TTSFeature
from api.inference.decisions.feature import DecisionsFeature
from api.memory_manager.queue import InferenceQueue, Job
from api.inference.decisions.model import decision_manager
from api.services.model_downloads import model_manager
from api.services.runtime import RuntimeFailure, runtime_manager
from api.inference.stt.model import get_transcription_manager
from api.services.video_jobs import video_jobs

log = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self, *, runtime=None, decisions=None, transcription=None, videos=None, queue=None):
        self.llm = LLMFeature(runtime or runtime_manager)
        self.video = VideoFeature(videos or video_jobs)
        self.image = ImageFeature()
        self.stt = STTFeature(transcription or get_transcription_manager())
        self.tts = TTSFeature()
        self.decisions = DecisionsFeature(decisions or decision_manager)
        self.features = {feature.name: feature for feature in
            (self.llm, self.video, self.image, self.stt, self.tts, self.decisions)}
        self.queue = queue or InferenceQueue(self._execute,
            url=os.getenv('KADAN_REDIS_URL', 'redis://127.0.0.1:6379/0'),
            lock_path=model_manager.root / 'inference.lock')

    @property
    def runtime(self):
        return self.llm.service

    @property
    def transcription(self):
        return self.stt.service

    @property
    def videos(self):
        return self.video.service

    @videos.setter
    def videos(self, service):
        self.video.service = service

    async def submit(self, body, *, feature, operation='generate'):
        wrapper = self.features[feature]
        model = wrapper.select(body)
        preflight = getattr(wrapper, 'preflight', None)
        if preflight is not None:
            preflight(body)
        job_id = await self.queue.submit(feature, operation, body.model_dump(mode='json'), model)
        queued = getattr(wrapper, 'queued', None)
        return queued(job_id, body) if queued is not None else await self.queue.wait(job_id)

    async def start(self):
        try:
            await self.queue.start()
            return True
        except RuntimeFailure:
            log.exception('Cannot start inference queue')
            return False

    async def close(self):
        try:
            await self.queue.close()
        finally:
            # ExitStack keeps attempting cleanup if any native close fails.
            async with AsyncExitStack() as cleanup:
                for feature in self.features.values():
                    cleanup.push_async_callback(feature.unload)

    async def video_job(self, job_id):
        job = self.videos.get(job_id)
        if job.status in ('Queued', 'Rendering'):
            try:
                record = await self.queue.get(job_id)
            except KeyError:
                record = {'state': 'failed', 'error': 'Inference job expired before completion.'}
            if record['state'] in ('failed', 'cancelled'):
                status = 'Failed' if record['state'] == 'failed' else 'Cancelled'
                self.videos._update(job_id, status=status, progress_text=status, error=record.get('error') or None)
                job = self.videos.get(job_id)
        return job

    async def _execute(self, job: Job):
        wrapper = self.features[job.feature]
        if job.operation not in wrapper.operations:
            raise RuntimeFailure('Unsupported inference operation.', 422)
        try:
            body = wrapper.validate(job.payload, job.operation)
        except ValidationError as exc:
            raise RuntimeFailure('Invalid queued inference payload.', 422) from exc
        if getattr(body, 'model', None) is not None and body.model != job.model:
            raise RuntimeFailure('Queued model selection does not match the request.', 422)
        # Each callable owns its heterogeneous request/result adaptation and its
        # atomic native load/restore/inference transaction. No model dispatch here.
        return await wrapper(body, model=job.model, operation=job.operation, job_id=job.id)


memory_manager = MemoryManager()
