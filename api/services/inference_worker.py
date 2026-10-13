"""Transport to the one inference owner; no Python inference scheduler."""

import asyncio
import json
import logging
import os
import signal
from collections import deque
from contextlib import suppress
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from api.pydantic_models.inference import (
    ImageProgress,
    InferenceProgress,
)
from api.pydantic_models.inference_task import InferenceTask
from api.pydantic_models.model_store import ModelStoreStatus
from api.services.inference_cancellation import await_cleanup
from api.services.inference_errors import InferenceFailure

log = logging.getLogger(__name__)


class JobState(StrEnum):
    QUEUED = "queued"
    PREPARING = "preparing"
    LOADING = "loading"
    RUNNING = "running"
    WAITING = "waiting_for_resources"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class WorkerMemory(BaseModel):
    capacity: list[int]
    used: list[int]
    residents: int


class WorkerEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = ""
    state: Literal[
        "ready",
        "models",
        "queued",
        "preparing",
        "loading",
        "running",
        "waiting_for_resources",
        "progress",
        "content",
        "succeeded",
        "failed",
        "cancelled",
        "rejected",
    ]
    result: JsonValue = None
    error: str = ""
    memory: WorkerMemory | None = None
    phase: str | None = None
    value: int = 0
    total: int = 0
    content: str | None = None
    protocol: int | None = None
    pid: int | None = None
    model: str | None = None
    context: dict[str, int | None] | None = None
    status_code: int = 502
    models: ModelStoreStatus | None = None


class WorkerExit(BaseModel):
    pid: int
    returncode: int
    signal: int | None = None
    stderr: str
    protocol_error: str | None = None

    def describe(self) -> str:
        cause = (
            f"signal {self.signal} ({signal.strsignal(self.signal)})"
            if self.signal
            else f"code {self.returncode}"
        )
        detail = self.stderr
        if self.protocol_error:
            detail = f"{self.protocol_error}: {detail}"
        return f"Inference worker {self.pid} exited with {cause}: {detail}"


class InferenceJob:
    """An outstanding response and bounded streaming channel, never an executor."""

    def __init__(self, request: InferenceTask, streaming: bool):
        self.request = request
        self.sent = False
        self.state = JobState.QUEUED
        self.accepted = asyncio.get_running_loop().create_future()
        self.result = asyncio.get_running_loop().create_future()
        self.model = None
        self.identified = asyncio.Event()
        self.stream_error = None
        self.image_progress = (
            ImageProgress(
                image_index=1,
                image_count=request.input.count,
                steps_total=request.input.steps,
            )
            if request.feature == "image"
            else None
        )
        self.progress: InferenceProgress | None = None
        self.events = asyncio.Queue(maxsize=64) if streaming else None


class InferenceWorker:
    def __init__(self, root: Path | None = None, on_event=None):
        self.root = root or (
            Path(os.getenv("KADAN_MODEL_DIR", "~/.local/share/kadan/models"))
            .expanduser()
            .resolve()
            / "inference"
        )
        self.on_event = on_event
        self.process = None
        self.reader = self.stderr_reader = None
        self.ready = None
        self.jobs: dict[str, InferenceJob] = {}
        self.diagnostics = deque(maxlen=64)
        self.memory: WorkerMemory | None = None
        self.models: ModelStoreStatus | None = None
        self.memory_observed_at: str | None = None
        self.last_exit: WorkerExit | None = None
        self.write_lock = asyncio.Lock()
        self.closing = False

    async def start(self):
        if self.process is not None:
            raise RuntimeError("Inference transport already started")
        self.root.mkdir(parents=True, exist_ok=True)
        executable = Path(
            os.getenv("KADAN_INFERENCE_WORKER", "/opt/kadan/bin/kadan-inference-worker")
        )
        if not executable.is_absolute() or not executable.is_file():
            raise InferenceFailure("The inference worker executable is unavailable.")
        self.ready = asyncio.get_running_loop().create_future()
        self.process = await asyncio.create_subprocess_exec(
            str(executable),
            str(self.root / "worker.lock"),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=128 * 1024 * 1024 + 1,
            start_new_session=True,
        )
        self.stderr_reader = asyncio.create_task(self._stderr())
        self.reader = asyncio.create_task(self._read())
        try:
            await asyncio.wait_for(asyncio.shield(self.ready), 30)
        except BaseException:
            await self.close()
            raise

    async def _send(self, message):
        if self.process is None or self.process.returncode is not None:
            raise InferenceFailure("Inference worker is not running.")
        encoded = (
            json.dumps(message, allow_nan=False, ensure_ascii=False) + "\n"
        ).encode()
        if len(encoded) > 3 * 1024 * 1024:
            raise InferenceFailure(
                "Inference request exceeds its transport limit.", 413
            )
        async with self.write_lock:
            self.process.stdin.write(encoded)
            if message["command"] == "submit":
                self.jobs[message["id"]].sent = True
            await asyncio.wait_for(self.process.stdin.drain(), 10)

    async def submit(self, request: InferenceTask, *, streaming=False) -> InferenceJob:
        if self.closing:
            raise InferenceFailure("Inference worker is stopping.")
        # Reject oversized input before registering a job that was never queued.
        message = request.model_dump(mode="json")
        if (
            len(
                (
                    json.dumps(message, allow_nan=False, ensure_ascii=False) + "\n"
                ).encode()
            )
            > 3 * 1024 * 1024
        ):
            raise InferenceFailure(
                "Inference request exceeds its transport limit.", 413
            )
        if request.id in self.jobs:
            raise InferenceFailure("Request is already active.", 409)
        job = InferenceJob(request, streaming)
        self.jobs[request.id] = job
        try:
            await self._send(message)
            await asyncio.wait_for(asyncio.shield(job.accepted), 10)
        except BaseException:
            if job.sent:
                with suppress(Exception, asyncio.CancelledError):
                    await await_cleanup(asyncio.create_task(self.cancel_and_join(job)))
            self.jobs.pop(request.id, None)
            raise
        return job

    async def cancel(self, job_id: str):
        await self._send(dict(command="cancel", id=job_id))

    async def cancel_and_join(self, job: InferenceJob):
        if not job.result.done():
            try:
                await self.cancel(job.request.id)
            except Exception:
                await self.close()
        with suppress(Exception):
            await asyncio.shield(job.result)

    async def _stderr(self):
        while data := await self.process.stderr.read(4096):
            self.diagnostics.append(data.decode("utf-8", errors="replace"))

    async def _read(self):
        protocol_error = None
        try:
            while line := await self.process.stdout.readline():
                event = WorkerEvent.model_validate_json(line)
                if event.state == "models":
                    if event.models is None:
                        raise RuntimeError("Worker model snapshot is missing")
                    self.models = event.models
                    continue
                if event.state == "ready":
                    if event.protocol != 2 or self.ready.done():
                        raise RuntimeError("Invalid worker readiness response")
                    self.ready.set_result(event.pid)
                    continue
                if event.memory is not None:
                    self.memory = event.memory
                    self.memory_observed_at = datetime.now(timezone.utc).isoformat()
                job = self.jobs.get(event.id)
                if job is None:
                    raise RuntimeError("Worker response has no outstanding request")
                if event.model is not None:
                    job.model = event.model
                    job.identified.set()
                if event.state in ("succeeded", "failed", "cancelled", "rejected"):
                    job.identified.set()
                if event.state == "content":
                    if job.events is not None and job.stream_error is None:
                        try:
                            job.events.put_nowait({"content": event.content})
                        except asyncio.QueueFull:
                            await self.cancel(event.id)
                            job.stream_error = "Chat stream consumer is too slow."
                    continue
                if event.state == "progress":
                    if self.on_event is not None:
                        self.on_event(job, event)
                    continue
                job.state = JobState(event.state)
                if not job.accepted.done():
                    if job.state == JobState.REJECTED:
                        job.accepted.set_exception(
                            InferenceFailure(
                                event.error, 429 if event.error == "queue_full" else 503
                            )
                        )
                    else:
                        job.accepted.set_result(None)
                if self.on_event is not None:
                    self.on_event(job, event)
                if job.state == JobState.SUCCEEDED:
                    job.result.set_result(event.result)
                elif job.state in (
                    JobState.FAILED,
                    JobState.CANCELLED,
                    JobState.REJECTED,
                ):
                    job.result.set_exception(
                        InferenceFailure(
                            event.error or "Inference cancelled.",
                            499
                            if job.state == JobState.CANCELLED
                            else event.status_code,
                        )
                    )
        except Exception as error:
            protocol_error = str(error)[:2000]
            if self.process.returncode is None:
                self.process.terminate()
        finally:
            try:
                returncode = await asyncio.wait_for(self.process.wait(), 30)
            except TimeoutError:
                os.killpg(self.process.pid, signal.SIGKILL)
                returncode = await self.process.wait()
            # A crashed worker may leave an owned codec process holding stderr open.
            if returncode != 0:
                with suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
            if self.stderr_reader is not None:
                await self.stderr_reader
            self.last_exit = WorkerExit(
                pid=self.process.pid,
                returncode=returncode,
                signal=-returncode if returncode < 0 else None,
                stderr="".join(self.diagnostics)[-8192:],
                protocol_error=protocol_error,
            )
            failure = InferenceFailure(self.last_exit.describe())
            if not self.ready.done():
                self.ready.set_exception(failure)
            for job in self.jobs.values():
                job.identified.set()
                if not job.accepted.done():
                    job.accepted.set_exception(failure)
                if not job.result.done():
                    job.result.set_exception(failure)
            if not self.closing:
                log.error("%s", failure)

    async def close(self):
        self.closing = True
        if self.process is None:
            return
        if self.process.returncode is None:
            self.process.stdin.close()
            try:
                await asyncio.wait_for(self.process.wait(), 30)
            except TimeoutError:
                os.killpg(self.process.pid, signal.SIGKILL)
                await self.process.wait()
        if self.reader is not None:
            await self.reader
        self.process = None
