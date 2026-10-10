"""One Redis consumer coordinating six concrete request executors."""
from contextlib import AsyncExitStack
import logging
import os

from pydantic import ValidationError

from api.inference.llm.chat_requests import ChatRequests
from api.inference.video.video_requests import VideoRequests
from api.inference.image.image_requests import ImageRequests
from api.inference.stt.transcription_requests import TranscriptionRequests
from api.inference.tts.speech_requests import SpeechRequests
from api.inference.decisions.decision_requests import DecisionRequests
from api.memory_manager.queue import InferenceQueue, Job
from api.memory_manager.streaming import QueuedStream
from api.inference.decisions.laya_backend import laya_evaluator
from api.services.model_downloads import model_manager
from api.inference.errors import InferenceFailure
from api.inference.progress import track
from api.services.chat_runtime import chat_runtime
from api.inference.stt.whisper_transcriber import get_whisper_transcriber
from api.services.video_jobs import video_jobs

log = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self, *, runtime=None, decisions=None, transcription=None, videos=None, queue=None):
        self.llm = ChatRequests(runtime or chat_runtime)
        self.video = VideoRequests(videos or video_jobs)
        self.image = ImageRequests()
        self.stt = TranscriptionRequests(transcription or get_whisper_transcriber())
        self.tts = SpeechRequests()
        self.decisions = DecisionRequests(decisions or laya_evaluator)
        self.request_executors = {feature.name: feature for feature in
            (self.llm, self.video, self.image, self.stt, self.tts, self.decisions)}
        self.queue = queue or InferenceQueue(self._execute,
            url=os.getenv('KADAN_REDIS_URL', 'redis://127.0.0.1:6379/0'),
            lock_path=model_manager.root / 'inference.lock')

    @property
    def runtime(self):
        return self.llm.chat_runtime

    @property
    def transcription(self):
        return self.stt.transcriber

    @property
    def videos(self):
        return self.video.video_jobs

    @videos.setter
    def videos(self, video_jobs):
        self.video.video_jobs = video_jobs

    async def submit(self, body, *, feature, operation='generate'):
        wrapper = self.request_executors[feature]
        model = wrapper.select(body)
        preflight = getattr(wrapper, 'preflight', None)
        if preflight is not None:
            preflight(body)
        job_id = await self.queue.submit(feature, operation, body.model_dump(mode='json'), model)
        queued = getattr(wrapper, 'queued', None)
        return queued(job_id, body) if queued is not None else await self.queue.wait(job_id)

    async def open_chat_stream(self, body):
        model = self.llm.select(body)
        stream = QueuedStream(self.queue)
        stream.job_id = await self.queue.submit("llm", "completion", body.model_dump(mode="json"), model, stream=stream)
        stream.start()
        return stream, model

    async def start(self):
        try:
            await self.queue.start()
            return True
        except InferenceFailure:
            log.exception('Cannot start inference queue')
            return False

    async def close(self):
        async def cleanup():
            # Attempt every executor; any failure retains the consumer lock.
            async with AsyncExitStack() as stack:
                for feature in self.request_executors.values():
                    stack.push_async_callback(feature.unload)
        await self.queue.close(cleanup=cleanup)

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
        with track(job.id, job.feature):
            return await self._execute_request(job)

    async def _execute_request(self, job: Job):
        wrapper = self.request_executors[job.feature]
        if job.operation not in wrapper.operations:
            raise InferenceFailure('Unsupported inference operation.', 422)
        try:
            body = wrapper.validate(job.payload, job.operation)
        except ValidationError as exc:
            raise InferenceFailure('Invalid queued inference payload.', 422) from exc
        if getattr(body, 'model', None) is not None and body.model != job.model:
            raise InferenceFailure('Queued model selection does not match the request.', 422)
        preflight_execution = getattr(wrapper, "preflight_execution", None)
        if preflight_execution is not None:
            await preflight_execution(body)
        # Unconfirmed cleanup from any feature blocks execution across the FIFO.
        # This checks ownership only; healthy residents and memory contention
        # retain their existing admission/eviction behavior.
        for feature in self.request_executors.values():
            check = getattr(feature, 'check_execution_state', None)
            if check is not None:
                check()
        # Each callable owns its heterogeneous request/result adaptation and its
        # atomic native load/restore/inference transaction. No model dispatch here.
        stream = self.queue.streams.get(job.id) if job.feature == "llm" else None
        return await wrapper(body, model=job.model, operation=job.operation, job_id=job.id,
            **({"on_event": stream.emit} if stream is not None else {}))


memory_manager = MemoryManager()
