"""API submission and presentation for the single native inference process."""
from __future__ import annotations
import asyncio
from contextlib import suppress
import logging
from pathlib import Path
import shutil
import tempfile
import time
from typing import TYPE_CHECKING

from api.inference.cancellation import await_cleanup, run_cancellable_thread
from api.inference.errors import InferenceFailure
from api.pydantic_models.inference import Feature, Operation, EmptyPayload
from api.services.inference_requests import prepare_request, present_result
from api.services.model_downloads import model_manager
from api.services.native_worker import JobState, NativeRequest, NativeWorker
from api.services.video_jobs import video_jobs
from api.services.whisper_selection import whisper_selection

if TYPE_CHECKING:
    from api.services.inference_requests import InferenceBody
    from api.routes.v1.chat.completions import CompletionRequest

log = logging.getLogger(__name__)


class NativeStream:
    def __init__(self, service, job, completion):
        self.service, self.job, self.completion = service, job, completion

    async def __aiter__(self):
        while not self.completion.done() or not self.job.events.empty():
            try:
                yield await asyncio.wait_for(self.job.events.get(), .25)
            except TimeoutError:
                continue
        result = await self.completion
        if self.job.stream_error:
            raise InferenceFailure(self.job.stream_error, 502)
        yield {key: result[key] for key in ('finish_reason', 'timing', 'cache') if key in result}

    async def aclose(self):
        if not self.completion.done():
            self.completion.cancel()
        with suppress(Exception, asyncio.CancelledError):
            await await_cleanup(self.completion)


class InferenceService:
    def __init__(self):
        self.worker = NativeWorker(model_manager.root / 'inference', self._observe)
        self.tasks: dict[str, asyncio.Task] = {}
        self.model_id = None
        self.state = 'unloaded'
        self.error = None
        self.context = {}
        self.progress = None

    async def start(self):
        await self.worker.start()

    async def close(self):
        try:
            for job_id in tuple(self.worker.jobs):
                await self.cancel(job_id)
        finally:
            await self.worker.close()

    def status(self):
        memory = self.worker.memory
        return dict(state=self.state, model_id=self.model_id, error=self.error,
            memory=memory.model_dump() if memory is not None else None, max_output_tokens=256,
            configured_context_limit=self.context.get('configured'),
            effective_context_limit=self.context.get('effective'), supported_context_limit=self.context.get('supported'))

    def _observe(self, job, event):
        feature = job.request.feature
        if event.state in ('preparing', 'loading', 'running', 'waiting_for_resources', 'progress'):
            self.progress = dict(job_id=job.request.id, workload=feature,
                stage=event.phase or event.state, value=event.value, observed_unix_ns=time.time_ns())
        elif event.state in ('succeeded', 'failed', 'cancelled'):
            if self.progress and self.progress['job_id'] == job.request.id:
                self.progress = None
        if event.state == 'loading':
            if feature == 'llm':
                self.model_id, self.state, self.error = job.request.model, 'loading', None
            elif self.state == 'ready':
                self.state = 'offloaded'
        if feature == 'llm':
            if event.state == 'succeeded':
                self.state = 'unloaded' if job.request.operation == 'unload' else 'ready'
                if self.state == 'unloaded':
                    self.model_id = None
                    self.context.clear()
            elif event.state in ('failed', 'cancelled'):
                self.state, self.error = 'error', event.error or 'Cancelled'
        if feature == 'video':
            if event.state in ('loading', 'running', 'progress'):
                video_jobs.update(job.request.id, status='Rendering', progress_text=event.phase or event.state)
            elif event.state in ('failed', 'cancelled'):
                video_jobs.update(job.request.id, status='Failed' if event.state == 'failed' else 'Cancelled',
                    progress_text=event.state, error=event.error or None)

    def select(self, body: InferenceBody, feature: Feature):
        if feature == 'llm':
            model = getattr(body, 'model_id', None) or getattr(body, 'model', None) or model_manager._read_selected_model_id()
            if model is None:
                raise InferenceFailure('Select a language model before submitting a request.', 422)
            return model
        if feature == 'image': return 'qwen-image-2.1'
        if feature == 'decisions': return 'laya'
        if feature == 'video': return body.model
        if feature == 'tts': return body.model_id
        if feature == 'stt': return body.model or whisper_selection.selected()
        raise InferenceFailure('Unsupported inference feature.', 422)

    async def _submit(self, body: InferenceBody, feature: Feature, operation: Operation, streaming: bool = False):
        model = self.select(body, feature)
        request = NativeRequest(feature=feature, operation=operation, model=model)
        if feature == 'video':
            video_jobs.prepare(request.id, body)
        try:
            job = await self.worker.submit(request, streaming=streaming)
        except BaseException:
            if feature == 'video':
                video_jobs.update(request.id, status='Failed', progress_text='Not accepted')
            raise
        task = asyncio.create_task(self._complete(job, body))
        self.tasks[request.id] = task
        task.add_done_callback(lambda completed: self.tasks.pop(request.id, None))
        return job, task

    async def submit(self, body: InferenceBody, *, feature: Feature, operation: Operation ='generate'):
        job, task = await self._submit(body, feature, operation)
        if feature == 'video':
            # Completion is reflected by the video metadata endpoint.
            task.add_done_callback(self._video_finished)
            return video_jobs.get(job.request.id)
        return await task

    @staticmethod
    def _video_finished(task):
        if not task.cancelled() and task.exception() is not None:
            log.warning('Video request failed: %s', task.exception())

    async def open_chat_stream(self, body: CompletionRequest):
        job, completion = await self._submit(body, 'llm', 'completion', streaming=True)
        return NativeStream(self, job, completion), job.request.model

    async def _complete(self, job, body: InferenceBody):
        request = job.request
        workspace = None
        try:
            preparation = asyncio.create_task(job.preparing.wait())
            try:
                done, _ = await asyncio.wait((preparation, job.result), return_when=asyncio.FIRST_COMPLETED)
                if job.result in done:
                    return await job.result
            finally:
                preparation.cancel()
            workspace = Path(tempfile.mkdtemp(prefix=request.id + '-', dir=self.worker.root))
            try:
                prepared = await run_cancellable_thread(prepare_request, request.feature,
                    request.operation, request.model, body, workspace)
            except Exception as error:
                await self.worker.prepared(job, EmptyPayload(), str(error))
                await asyncio.shield(job.result)
                raise
            if request.feature == 'llm' and request.operation != 'unload':
                configured = (body.context_limit if request.operation == 'load' and 'context_limit' in body.model_fields_set
                              else model_manager.configured_context(request.model))
                self.context = dict(configured=configured, effective=prepared.context_limit,
                                    supported=prepared.supported_context)
            await self.worker.prepared(job, prepared.payload)
            result = await asyncio.shield(job.result)
            if request.operation == 'load':
                model_manager.select(request.model)
                if 'context_limit' in body.model_fields_set:
                    model_manager.set_context(request.model, body.context_limit)
                return self.status()
            if request.operation == 'unload':
                return self.status()
            if request.feature == 'video':
                published = await await_cleanup(asyncio.create_task(asyncio.to_thread(
                    video_jobs.publish, request.id, Path(result['output']), workspace)))
                return published.model_dump(mode='json')
            return await await_cleanup(asyncio.create_task(asyncio.to_thread(
                present_result, request.feature, request.model, body,
                prepared, result, workspace, request.id)))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if request.feature == 'video':
                video_jobs.update(request.id, status='Failed', progress_text='Failed', error=str(error))
            raise
        finally:
            try:
                if not job.result.done():
                    await await_cleanup(asyncio.create_task(self.worker.cancel_and_join(job)))
            finally:
                self.worker.jobs.pop(request.id, None)
                if workspace is not None:
                    shutil.rmtree(workspace)

    async def cancel(self, job_id):
        if job_id not in self.worker.jobs:
            return
        task = self.tasks.get(job_id)
        if task is not None:
            task.cancel()
            with suppress(Exception, asyncio.CancelledError):
                await await_cleanup(task)
        # A task cancelled before its first step never enters _complete's finally.
        job = self.worker.jobs.get(job_id)
        if job is not None:
            if not job.result.done():
                await await_cleanup(asyncio.create_task(self.worker.cancel_and_join(job)))
            self.worker.jobs.pop(job_id, None)


    async def unload(self):
        for job in tuple(self.worker.jobs.values()):
            if job.request.feature == 'llm':
                await self.cancel(job.request.id)
        request = NativeRequest(feature='llm', operation='unload', model=self.model_id or 'none')
        job = await self.worker.submit(request)
        try:
            await self.worker.prepared(job, EmptyPayload())
            await asyncio.shield(job.result)
            return self.status()
        finally:
            try:
                if not job.result.done():
                    await await_cleanup(asyncio.create_task(self.worker.cancel_and_join(job)))
            finally:
                self.worker.jobs.pop(request.id, None)


inference = InferenceService()
