"""Queue submission, response streaming, and observed worker status."""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from contextlib import suppress

from api.pydantic_models.inference import (
    Feature,
    InferenceProgress,
    Operation,
)
from api.pydantic_models.inference_task import (
    AudioCallTask,
    DownloadTask,
    InferenceTask,
    VideoTask,
)
from api.pydantic_models.model_store import (
    DownloadObservation,
    ModelStoreStatus,
)
from api.services.inference_cancellation import await_cleanup
from api.services.inference_errors import InferenceFailure
from api.services.video_jobs import video_jobs
from api.services.worker_hub import WorkerHub
from kadan_tracing import span, traced
from pydantic import BaseModel

log = logging.getLogger(__name__)


class InferenceStream:
    def __init__(self, service, job, completion):
        self.service, self.job, self.completion = service, job, completion

    async def __aiter__(self):
        while not self.completion.done() or not self.job.events.empty():
            try:
                yield await asyncio.wait_for(self.job.events.get(), 0.25)
            except TimeoutError:
                continue
        result = await self.completion
        if self.job.stream_error:
            raise InferenceFailure(self.job.stream_error, 502)
        if self.job.request.feature == "audio_call":
            yield {"type": "reply", **result}
        else:
            yield {
                key: result[key]
                for key in ("finish_reason", "timing", "cache")
                if key in result
            }

    async def aclose(self):
        await await_cleanup(
            asyncio.create_task(self.service.cancel(self.job.request.id))
        )
        with suppress(Exception, asyncio.CancelledError):
            await await_cleanup(self.completion)


class InferenceService:
    def __init__(self):
        self.worker = WorkerHub(on_event=self._observe)
        self.tasks: dict[str, asyncio.Task] = {}
        self.model_id = None
        self.state = "unloaded"
        self.error = None
        self.context = {}
        self.progress: InferenceProgress | None = None
        self.downloads: dict[str, DownloadObservation] = {}
        self._download_admissions: set[str] = set()

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
        state, model_id = self.state, self.model_id
        error = self.error
        context = {} if state == "error" else self.context
        loading = any(
            job.request.feature == "llm"
            and job.model == model_id
            and job.state in ("loading", "running", "waiting_for_resources")
            and not job.result.done()
            and any(worker_id in self.worker.workers for worker_id in job.machines)
            for job in self.worker.jobs.values()
        )
        if state in ("ready", "offloaded") or (state == "loading" and not loading):
            resident = next(
                (
                    worker
                    for worker in self.worker.workers.values()
                    if worker.kind == "llm"
                    and worker.residency_known
                    and worker.resident_model == model_id
                    and model_id is not None
                ),
                None,
            )
            state = "ready" if resident is not None else "unloaded"
            model_id = resident.resident_model if resident is not None else None
            context = resident.resident_context if resident is not None else {}
            if resident is None and any(
                worker.kind == "llm" and not worker.residency_known
                for worker in self.worker.workers.values()
            ):
                state, model_id = "error", self.model_id
                error = "Language-model residency is unconfirmed after a failed or cancelled request."
        return {
            "state": state,
            "model_id": model_id,
            "error": error,
            "memory": memory.model_dump() if memory is not None else None,
            "max_output_tokens": 256,
            "configured_context_limit": self.context.get("configured"),
            "effective_context_limit": context.get("effective"),
            "supported_context_limit": self.context.get("supported"),
        }

    def language_status(self):
        model_id = "lfm3-1.7b-bf16"
        checkpoint = (
            next(
                (model for model in self.worker.models.models if model.id == model_id),
                None,
            )
            if self.worker.models
            else None
        )
        workers = [
            worker
            for worker in self.worker.workers.values()
            if worker.kind == "lfm3_llm"
        ]
        error = None
        if not workers or checkpoint is None:
            error = "The LFM3 conversation worker is unavailable."
        elif checkpoint.status != "complete":
            error = (
                "Download the language checkpoint in Hikari Settings before inference."
            )
        elif any(not worker.residency_known for worker in workers):
            error = "Language-model cleanup is unconfirmed."
        active = [
            job
            for job in self.worker.jobs.values()
            if job.worker == "lfm3_llm" and not job.result.done()
        ]
        running = any(job.state == "running" for job in active)
        state = (
            "error"
            if error
            else "ready"
            if running
            else "loading"
            if active
            else "unloaded"
        )
        return {
            "state": state,
            "model_id": model_id,
            "error": error,
            "memory": self.worker.memory.model_dump() if self.worker.memory else None,
            "configured_context_limit": checkpoint.context_limit
            if checkpoint
            else None,
            "effective_context_limit": None,
            "supported_context_limit": checkpoint.architecture_context_limit
            if checkpoint
            else None,
            "max_output_tokens": None,
        }

    def models_status(self) -> ModelStoreStatus:
        if self.worker.models is None:
            raise InferenceFailure("Worker model status is unavailable.", 503)
        snapshot = self.worker.models.model_copy(deep=True)
        for model in snapshot.models:
            observation = self.downloads.get(model.id)
            if observation is not None and model.status != "complete":
                for field in DownloadObservation.model_fields:
                    setattr(model, field, getattr(observation, field))
        return snapshot

    async def download(self, model_id: str, license_acknowledged: bool):
        if not license_acknowledged:
            raise InferenceFailure(
                "Review and acknowledge model licenses before downloading.", 422
            )
        if not any(model.id == model_id for model in self.models_status().models):
            raise InferenceFailure(
                "The checkpoint is unavailable from connected workers.", 422
            )
        if model_id in self._download_admissions or any(
            job.request.feature == "models"
            and job.request.input.model_id == model_id
            and not job.result.done()
            for job in self.worker.jobs.values()
        ):
            raise InferenceFailure("A download is already queued for this model.", 409)
        observation = DownloadObservation()
        self.downloads[model_id] = observation
        self._download_admissions.add(model_id)
        try:
            _, completion = await self._submit(
                DownloadTask(
                    model_id=model_id, license_acknowledged=license_acknowledged
                ),
                "models",
                "download",
            )
        except BaseException:
            self.downloads[model_id] = DownloadObservation(
                status="failed", error="Download was not queued"
            )
            raise
        finally:
            self._download_admissions.discard(model_id)

        def finished(task):
            error = None if task.cancelled() else task.exception()
            # Keep a late failure attached to this download, not a newer retry.
            if error is not None and observation.status in (
                "downloading",
                "cancelling",
            ):
                observation.status, observation.error = "failed", str(error)
            self._background_finished(task)

        completion.add_done_callback(finished)
        return self.models_status()

    async def cancel_download(self, model_id: str):
        for job in self.worker.jobs.values():
            if (
                job.request.feature == "models"
                and job.request.input.model_id == model_id
                and not job.result.done()
            ):
                self.downloads[model_id].status = "cancelling"
                await self.worker.cancel(job.request.id)
                return self.models_status()
        raise InferenceFailure("No active download for this model.", 409)

    def _observe(self, job, event):
        if job.request.feature == "models":
            observation = self.downloads.setdefault(
                job.request.input.model_id, DownloadObservation()
            )
            if event.phase == "download":
                observation.downloaded_bytes, observation.total_bytes = (
                    event.value,
                    event.total,
                )
            elif event.state in ("succeeded", "failed", "cancelled", "rejected"):
                observation.status = (
                    "complete"
                    if event.state == "succeeded"
                    else "failed"
                    if event.state == "rejected"
                    else event.state
                )
                observation.error = event.error or None
            return
        if job.request.operation == "configure":
            return
        feature = job.request.feature
        native_language = feature == "llm" and job.worker != "lfm3_llm"
        if native_language and event.context is not None:
            self.context = event.context
        if native_language and event.model is not None:
            self.model_id = event.model
        image = job.image_progress
        if image is not None:
            if event.state in ("preparing", "waiting_for_resources", "loading"):
                image.phase = "loading" if event.state == "loading" else "queued"
            elif event.state == "running":
                image.phase = "prompt"
            elif event.state == "progress":
                if event.phase == "prepared":
                    image.steps_total = event.total
                elif event.phase == "image_started":
                    image.phase = "prompt"
                    image.image_index = event.value
                    image.steps_completed = 0
                elif event.phase == "step_completed":
                    image.phase = "denoise"
                    image.steps_completed = event.value
                elif event.phase in ("conditioning", "block_started", "block", "qkv"):
                    image.phase = "denoise"
                elif event.phase == "vae_up_block":
                    image.phase = "decode"
        if event.state in (
            "queued",
            "preparing",
            "loading",
            "running",
            "waiting_for_resources",
            "progress",
        ):
            job.progress = InferenceProgress(
                job_id=job.request.id,
                workload=feature,
                stage=event.phase or event.state,
                value=event.value,
                observed_unix_ns=time.time_ns(),
                image=image.model_copy() if image is not None else None,
            )
            if event.state != "queued":
                self.progress = job.progress
        elif event.state in ("succeeded", "failed", "cancelled", "rejected"):
            job.progress = None
            if self.progress and self.progress.job_id == job.request.id:
                self.progress = None
        if event.state == "loading" and native_language:
            self.model_id, self.state, self.error = (
                event.model or self.model_id,
                "loading",
                None,
            )
        if native_language:
            if event.state == "succeeded":
                self.state = (
                    "unloaded" if job.request.operation == "unload" else "ready"
                )
                if self.state == "unloaded":
                    self.model_id = None
                    self.context.clear()
            elif event.state in ("failed", "cancelled"):
                self.state, self.error = "error", event.error or "Cancelled"
        if feature == "video":
            if event.state in ("loading", "running", "progress"):
                video_jobs.update(
                    job.request.id,
                    status="Rendering",
                    progress_text=event.phase or event.state,
                )
            elif event.state in ("failed", "cancelled"):
                video_jobs.update(
                    job.request.id,
                    status="Failed" if event.state == "failed" else "Cancelled",
                    progress_text=event.state,
                    error=event.error or None,
                )

    async def _submit(
        self,
        body: BaseModel,
        feature: Feature,
        operation: Operation,
        streaming: bool = False,
        job_id: str | None = None,
    ):
        request = InferenceTask(
            **({"id": job_id} if job_id is not None else {}),
            operation=operation,
            input={
                "feature": feature,
                **body.model_dump(mode="json", exclude_unset=True),
            },
        )
        if isinstance(request.input, VideoTask):
            video = request.input
            if video.duration is None or video.resolution is None:
                status = self.models_status()
                if video.duration is None:
                    video.duration = status.video_defaults.duration
                if video.resolution is None:
                    video.resolution = status.video_defaults.resolution
            video_jobs.prepare(request.id, video)
        try:
            job = await self.worker.submit(request, streaming=streaming)
        except BaseException:
            if feature == "video":
                video_jobs.update(
                    request.id, status="Failed", progress_text="Not accepted"
                )
            raise
        task = asyncio.create_task(self._complete(job))
        self.tasks[request.id] = task
        task.add_done_callback(lambda completed: self.tasks.pop(request.id, None))
        return job, task

    async def submit(
        self,
        body: BaseModel,
        *,
        feature: Feature,
        operation: Operation = "generate",
        job_id: str | None = None,
    ):
        job, task = await self._submit(body, feature, operation, job_id=job_id)
        if feature == "video":
            # Completion is reflected by the video metadata endpoint.
            task.add_done_callback(self._background_finished)
            return video_jobs.get(job.request.id)
        try:
            return await task
        except asyncio.CancelledError:
            await await_cleanup(asyncio.create_task(self.cancel(job.request.id)))
            raise

    @staticmethod
    def _background_finished(task):
        if not task.cancelled() and task.exception() is not None:
            log.warning("Background request failed: %s", task.exception())

    async def open_audio_turn(self, body: AudioCallTask):
        if body.mode != "turn":
            raise InferenceFailure("Audio streaming requires a complete turn.", 422)
        job, completion = await self._submit(
            body, "audio_call", "generate", streaming=True
        )
        return InferenceStream(self, job, completion)

    async def open_chat_stream(self, body: BaseModel):
        job, completion = await self._submit(body, "llm", "completion", streaming=True)
        try:
            # Use the worker's selected model in every SSE chunk.
            await job.identified.wait()
            if job.model is None:
                await completion
                raise InferenceFailure("Worker did not identify the chat model.", 502)
            return InferenceStream(self, job, completion), job.model
        except BaseException:
            await await_cleanup(asyncio.create_task(self.cancel(job.request.id)))
            raise

    @traced("kadan.inference.wait_and_deliver")
    async def _complete(self, job):
        failure = None
        try:
            result = await asyncio.shield(job.result)
            with span("kadan.result.delivery"):
                if job.request.feature == "video":
                    # The worker may be on another machine; its file arrives inline.
                    video = result.pop("video", None)
                    if video is not None:
                        await asyncio.to_thread(
                            video_jobs.store, job.request.id, base64.b64decode(video)
                        )
                        result["output_url"] = f"/v1/videos/{job.request.id}/content"
                    video_jobs.update(job.request.id, **result)
                if job.request.operation in ("load", "unload"):
                    return self.status()
                return result
        except (Exception, asyncio.CancelledError) as error:
            failure = error
            raise
        finally:
            try:
                if not job.result.done():
                    await await_cleanup(
                        asyncio.create_task(self.worker.cancel_and_join(job))
                    )
            finally:
                if self.progress and self.progress.job_id == job.request.id:
                    self.progress = None
                self.worker.jobs.pop(job.request.id, None)
                if failure is not None and job.result.done():
                    self._finish_video_error(job, failure)

    @staticmethod
    def _finish_video_error(job, error):
        if job.request.feature != "video":
            return
        cancelled = isinstance(error, asyncio.CancelledError) or (
            isinstance(error, InferenceFailure) and error.status_code == 499
        )
        video_jobs.finish_error(
            job.request.id, cancelled=cancelled, error=None if cancelled else str(error)
        )

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
            try:
                if not job.result.done():
                    await await_cleanup(
                        asyncio.create_task(self.worker.cancel_and_join(job))
                    )
            finally:
                self.worker.jobs.pop(job_id, None)
                if job.result.done():
                    error = None if job.result.cancelled() else job.result.exception()
                    self._finish_video_error(job, error or asyncio.CancelledError())

    async def unload(self):
        """Cancel language requests and await confirmed cold-worker cleanup."""
        for job in tuple(self.worker.jobs.values()):
            if job.request.feature == "llm":
                await self.cancel(job.request.id)
        await self.worker.offload("lfm3_llm")
        return self.language_status()


inference = InferenceService()
