"""Transport to the one native inference owner; no Python inference scheduler."""
import asyncio
from collections import deque
from contextlib import suppress
from enum import StrEnum
import json
import logging
import os
from pathlib import Path
import signal
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from api.inference.errors import InferenceFailure
from api.pydantic_models.inference import Feature, Operation, NativePayload, EmptyPayload
from api.inference.cancellation import await_cleanup

log = logging.getLogger(__name__)


class JobState(StrEnum):
    QUEUED = 'queued'
    PREPARING = 'preparing'
    LOADING = 'loading'
    RUNNING = 'running'
    WAITING = 'waiting_for_resources'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'
    CANCELLED = 'cancelled'
    REJECTED = 'rejected'


class NativeRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(default_factory=lambda: uuid4().hex, pattern=r'^[a-f0-9]{32}$')
    command: Literal['submit'] = 'submit'
    feature: Feature
    operation: Operation
    model: str = Field(min_length=1, max_length=4096)


class NativeMemory(BaseModel):
    capacity: list[int]
    used: list[int]
    residents: int


class NativeEvent(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = ''
    state: Literal['ready', 'queued', 'preparing', 'loading', 'running', 'waiting_for_resources',
                   'progress', 'content', 'succeeded', 'failed', 'cancelled', 'rejected']
    result: JsonValue = None
    error: str = ''
    memory: NativeMemory | None = None
    phase: str | None = None
    value: int = 0
    content: str | None = None
    protocol: int | None = None
    pid: int | None = None


class NativeJob:
    """An outstanding response and bounded streaming channel, never an executor."""
    def __init__(self, request: NativeRequest, streaming: bool):
        self.request = request
        self.state = JobState.QUEUED
        self.preparing = asyncio.Event()
        self.prepared_sent = False
        self.accepted = asyncio.get_running_loop().create_future()
        self.result = asyncio.get_running_loop().create_future()
        self.stream_error = None
        self.events = asyncio.Queue(maxsize=64) if streaming else None


class NativeWorker:
    def __init__(self, root: Path, on_event=None):
        self.root = root
        self.on_event = on_event
        self.process = None
        self.reader = self.stderr_reader = None
        self.ready = None
        self.jobs: dict[str, NativeJob] = {}
        self.diagnostics = deque(maxlen=64)
        self.memory: NativeMemory | None = None
        self.write_lock = asyncio.Lock()
        self.closing = False

    async def start(self):
        if self.process is not None:
            raise RuntimeError('Native inference transport already started')
        self.root.mkdir(parents=True, exist_ok=True)
        executable = Path(os.getenv('KADAN_INFERENCE_WORKER', '/opt/kadan/bin/kadan-inference-worker'))
        if not executable.is_absolute() or not executable.is_file():
            raise InferenceFailure('The native inference worker executable is unavailable.')
        self.ready = asyncio.get_running_loop().create_future()
        self.process = await asyncio.create_subprocess_exec(
            str(executable), str(self.root / 'worker.lock'),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, limit=1024 * 1024 + 1, start_new_session=True)
        self.stderr_reader = asyncio.create_task(self._stderr())
        self.reader = asyncio.create_task(self._read())
        try:
            await asyncio.wait_for(asyncio.shield(self.ready), 30)
        except BaseException:
            await self.close()
            raise

    async def _send(self, message):
        if self.process is None or self.process.returncode is not None:
            raise InferenceFailure('Native inference worker is not running.')
        encoded = (json.dumps(message, allow_nan=False, ensure_ascii=False) + '\n').encode()
        if len(encoded) > 256 * 1024:
            raise InferenceFailure('Native request exceeds its transport limit.', 413)
        async with self.write_lock:
            self.process.stdin.write(encoded)
            await asyncio.wait_for(self.process.stdin.drain(), 10)

    async def submit(self, request: NativeRequest, *, streaming=False) -> NativeJob:
        if self.closing:
            raise InferenceFailure('Native inference worker is stopping.')
        job = NativeJob(request, streaming)
        self.jobs[request.id] = job
        try:
            await self._send(request.model_dump())
            await asyncio.wait_for(asyncio.shield(job.accepted), 10)
        except BaseException:
            with suppress(Exception, asyncio.CancelledError):
                await await_cleanup(asyncio.create_task(self.cancel_and_join(job)))
            self.jobs.pop(request.id, None)
            raise
        return job

    async def prepared(self, job: NativeJob, payload: NativePayload, error: str = ''):
        sending = asyncio.create_task(self._send(dict(command='prepared', id=job.request.id, payload=payload.model_dump(mode='json', by_alias=True, exclude_none=True), error=error[:2000])))
        try:
            await await_cleanup(sending)
        finally:
            if sending.done() and not sending.cancelled() and sending.exception() is None:
                job.prepared_sent = True

    async def cancel(self, job_id: str):
        await self._send(dict(command='cancel', id=job_id))

    async def cancel_and_join(self, job: NativeJob):
        if not job.result.done():
            try:
                await self.cancel(job.request.id)
                if not job.prepared_sent:
                    await self.prepared(job, EmptyPayload(), 'Preparation cancelled')
            except Exception:
                await self.close()
        with suppress(Exception):
            await asyncio.shield(job.result)

    async def _stderr(self):
        while data := await self.process.stderr.read(4096):
            self.diagnostics.append(data.decode('utf-8', errors='replace'))

    async def _read(self):
        try:
            while line := await self.process.stdout.readline():
                event = NativeEvent.model_validate_json(line)
                if event.state == 'ready':
                    if event.protocol != 1 or self.ready.done():
                        raise RuntimeError('Invalid native readiness response')
                    self.ready.set_result(event.pid)
                    continue
                if event.memory is not None:
                    self.memory = event.memory
                job = self.jobs.get(event.id)
                if job is None:
                    raise RuntimeError('Native response has no outstanding request')
                if event.state == 'content':
                    if job.events is not None and job.stream_error is None:
                        try:
                            job.events.put_nowait({'content': event.content})
                        except asyncio.QueueFull:
                            await self.cancel(event.id)
                            job.stream_error = 'Chat stream consumer is too slow.'
                    continue
                if event.state == 'progress':
                    if self.on_event is not None:
                        self.on_event(job, event)
                    continue
                job.state = JobState(event.state)
                if job.state == JobState.PREPARING:
                    job.preparing.set()
                if not job.accepted.done():
                    if job.state == JobState.REJECTED:
                        job.accepted.set_exception(InferenceFailure(event.error, 429 if event.error == 'queue_full' else 503))
                    else:
                        job.accepted.set_result(None)
                if self.on_event is not None:
                    self.on_event(job, event)
                if job.state == JobState.SUCCEEDED:
                    job.result.set_result(event.result)
                elif job.state in (JobState.FAILED, JobState.CANCELLED, JobState.REJECTED):
                    job.result.set_exception(InferenceFailure(event.error or 'Inference cancelled.',
                        499 if job.state == JobState.CANCELLED else 502))
        except Exception as error:
            self.diagnostics.append(str(error))
            if self.process.returncode is None:
                self.process.terminate()
        finally:
            try:
                returncode = await asyncio.wait_for(self.process.wait(), 30)
            except TimeoutError:
                os.killpg(self.process.pid, signal.SIGKILL)
                returncode = await self.process.wait()
            if self.stderr_reader is not None:
                await self.stderr_reader
            detail = ''.join(self.diagnostics)[-8192:]
            failure = InferenceFailure(f'Native worker exited with status {returncode}: {detail}')
            if not self.ready.done():
                self.ready.set_exception(failure)
            for job in self.jobs.values():
                if not job.accepted.done():
                    job.accepted.set_exception(failure)
                if not job.result.done():
                    job.result.set_exception(failure)
            if not self.closing:
                log.error('%s', failure)

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
