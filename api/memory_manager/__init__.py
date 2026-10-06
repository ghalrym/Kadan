"""One callable per inference feature; one in-process Redis consumer.

ResourceManager remains the only RAM/VRAM owner. Feature objects delegate to the
selected native adapters, whose existing leases and restore hooks protect work.
"""
import asyncio
from contextlib import suppress
import logging
import os
import threading

from pydantic import ValidationError

from api.inference.resources import ResourceBusy, ResourceExhausted
from api.inference.video import VideoSpec
from api.memory_manager.queue import InferenceQueue, Job
from api.pydantic_models.chat import ChatMessage
from api.services.decisions import decision_manager
from api.services.model_downloads import model_manager
from api.services.runtime import RuntimeFailure, finish_cleanup, runtime_manager
from api.services.transcription.transcription import get_transcription_manager
from api.services.video_jobs import video_jobs

log = logging.getLogger(__name__)


async def native_call(function, *args):
    """Cancel cooperatively and wait for the native thread before advancing FIFO."""
    cancel = threading.Event()
    worker = asyncio.create_task(asyncio.to_thread(function, *args, cancel))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancel.set()
        with suppress(Exception, asyncio.CancelledError):
            await finish_cleanup(worker)
        raise


class Feature:
    def __init__(self, manager, name):
        self.manager, self.name = manager, name

    async def __call__(self, body, *, operation='generate'):
        payload = body.model_dump(mode='json')
        model = payload.get('model')
        if self.name == 'llm':
            model = model or self.manager.runtime.model_id
        elif self.name == 'stt':
            model = model or self.manager.transcription.selected()
        elif self.name == 'decisions':
            model = 'laya'
        if self.name == 'video':
            self.manager.videos.validate(model, VideoSpec(**body.model_dump(exclude={'model'})))
        job_id = await self.manager.queue.submit(self.name, operation, payload, model)
        if self.name == 'video':
            return self.manager.videos.prepare(job_id, VideoSpec(**body.model_dump(exclude={'model'})))
        return await self.manager.queue.wait(job_id)


class MemoryManager:
    def __init__(self, *, runtime=None, decisions=None, transcription=None, videos=None, queue=None):
        self.runtime = runtime or runtime_manager
        self.decision_service = decisions or decision_manager
        self._transcription = transcription
        self.videos = videos or video_jobs
        self.llm = Feature(self, 'llm')
        self.video = Feature(self, 'video')
        self.image = Feature(self, 'image')
        self.stt = Feature(self, 'stt')
        self.tts = Feature(self, 'tts')
        self.decisions = Feature(self, 'decisions')
        self.queue = queue or InferenceQueue(self._execute,
            url=os.getenv('KADAN_REDIS_URL', 'redis://127.0.0.1:6379/0'),
            lock_path=model_manager.root / 'inference.lock')

    @property
    def transcription(self):
        if self._transcription is None:
            self._transcription = get_transcription_manager()
        return self._transcription

    async def start(self):
        try:
            await self.queue.start()
            return True
        except RuntimeFailure:
            # Settings, downloads and health stay available while inference is down.
            log.exception('Cannot start inference queue')
            return False

    async def close(self):
        try:
            await self.queue.close()
        finally:
            if self._transcription is not None:
                await asyncio.to_thread(self._transcription.close)

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

    def _handoff(self, workload, cancel):
        # First CUDA discovery/import can take seconds. Keep it off the event
        # loop so Redis deadlines and client disconnects remain responsive.
        self.runtime.ensure_resources().offload_inactive_devices(workload, cancel)

    async def _execute(self, job: Job):
        # Route models own HTTP validation. Import at dispatch to avoid the
        # routes -> shared manager -> routes construction cycle, and revalidate
        # durable JSON before it can reach a native adapter.
        from api.routes.v1.chat.completions import CompletionRequest
        from api.routes.v1.decisions import DecisionRequest
        from api.routes.v1.audio.transcriptions import TranscriptionRequest
        from api.routes.v1.audio.speech import SpeechRequest
        from api.routes.v1.images.generations import ImageRequest
        from api.routes.v1.images.edits import ImageRequest as EditRequest
        from api.routes.v1.videos.generations import VideoGenerationRequest

        validators = {'llm': CompletionRequest, 'decisions': DecisionRequest,
            'stt': TranscriptionRequest, 'tts': SpeechRequest, 'video': VideoGenerationRequest,
            'image': EditRequest if job.operation == 'edit' else ImageRequest}
        if job.operation not in (('generate', 'edit') if job.feature == 'image' else ('generate',)):
            raise RuntimeFailure('Unsupported inference operation.', 422)
        try:
            body = validators[job.feature].model_validate(job.payload)
        except ValidationError as exc:
            raise RuntimeFailure('Invalid queued inference payload.', 422) from exc
        if getattr(body, 'model', None) is not None and body.model != job.model:
            raise RuntimeFailure('Queued model selection does not match the request.', 422)
        if job.feature == 'llm' and job.model is None:
            raise RuntimeFailure('No model is ready. Load a model in Settings.')
        if job.feature == 'image':
            raise RuntimeFailure('No image provider is configured. Image generation and editing are unavailable.')
        if job.feature == 'tts':
            raise RuntimeFailure('No speech provider is configured. Speech generation and voice cloning are unavailable.')
        workload = {'llm': 'llm', 'video': 'video', 'stt': 'speech', 'decisions': 'decision'}[job.feature]
        try:
            # Native callbacks offload only supported device allocations. LLM
            # expert banks and CPU Decisions remain host residents; RAM pressure
            # can still evict them through their existing host callbacks.
            await native_call(self._handoff, workload)
        except (ResourceBusy, ResourceExhausted) as exc:
            raise RuntimeFailure(str(exc)) from exc
        if job.feature == 'llm':
            return await self.runtime.complete([ChatMessage.model_validate(m) for m in job.payload['messages']], job.model)
        if job.feature == 'decisions':
            result = await self.decision_service.evaluate(body.state, body.questions)
            return [answer.model_dump(mode='json') if hasattr(answer, 'model_dump') else answer for answer in result]
        if job.feature == 'stt':
            result = await native_call(self.transcription.transcribe, body.audio, job.model, body.language)
            result['formatting_status'] = 'unavailable' if body.formatting else 'disabled'
            return result
        spec = VideoSpec(**body.model_dump(exclude={'model'}))
        return await native_call(self.videos.run, job.id, job.model, spec)


memory_manager = MemoryManager()
