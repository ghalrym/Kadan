"""Connected worker containers, job routing, and memory scheduling across them.

Every worker container runs one program for one inference type and connects
here on its own. Workers on the same computer share its RAM and GPUs, so the
hub keeps one ledger per computer (identified by its kernel boot ID) with each
GPU counted once (identified by its UUID). The hub owns every memory decision:
a worker only pulls jobs while the hub has granted it a budget.
"""

import asyncio
import hmac
import json
import logging
import os
import time
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from uuid import uuid4

from api.pydantic_models.inference_task import InferenceTask
from api.pydantic_models.model_store import ModelStoreStatus
from api.services.inference_cancellation import await_cleanup
from api.services.inference_errors import InferenceFailure
from api.services.inference_worker import (
    TERMINAL,
    InferenceJob,
    JobState,
    WorkerEvent,
    WorkerMemory,
    encode_request,
)
from api.services.memory_scheduler import (
    GIB,
    PRIORITY,
    TOP_PRIORITY,
    Action,
    MachineLedger,
    Requirement,
    WorkerSlot,
    needs_grant,
    plan_growth,
    plan_load,
    releases,
)
from api.services.tracing import JobTracing
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)

PROTOCOL = 4

# Features served by an inference worker; settings go to every worker and a
# download goes to a worker of its checkpoint's type.
FEATURE_WORKERS = {
    "llm": "llm",
    "decisions": "decision",
    "tts": "tts",
    "stt": "speech",
    "stt_formatting": "formatting",
    "image": "image",
    "video": "video",
}

# A checkpoint's catalog kind and the worker type that downloads and runs it.
CHECKPOINT_WORKERS = {
    "llm": "llm",
    "decision": "decision",
    "speech": "tts",
    "transcription": "speech",
    "formatting": "formatting",
    "image": "image",
    "video": "video",
    "vision": "lfm3_vision",
    "audio": "lfm3_audio",
}

EXPERIMENTAL_MODELS = {
    "lfm3-1.7b-bf16": ("lfm3_llm", "HIKARI_ENABLE_LFM3_TEXT"),
    "lfm3-vl-2b-bf16": ("lfm3_vision", "HIKARI_ENABLE_LFM3_VISION"),
    "lfm3-audio-2b-bf16": ("lfm3_audio", "HIKARI_ENABLE_LFM3_AUDIO"),
}
EXPERIMENTAL_FEATURES = {
    "vision": "lfm3-vl-2b-bf16",
    "audio_call": "lfm3-audio-2b-bf16",
}


def experimental_worker(model: str | None) -> str | None:
    if model not in EXPERIMENTAL_MODELS:
        return None
    kind, flag = EXPERIMENTAL_MODELS[model]
    if os.getenv(flag, "").lower() not in ("1", "true", "yes"):
        raise InferenceFailure(f"Experimental model requires {flag}=true.", 409)
    return kind


# Jobs that have not started executing can move to another worker.
REQUEUEABLE = (JobState.QUEUED, JobState.PREPARING, JobState.WAITING)

# A release the worker never confirms is reread from its status after this.
RELEASE_TIMEOUT = 60

# How long a request waits for a worker of its type to connect.
CONNECT_TIMEOUT = 30


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HIKARI_", env_file=".env", extra="ignore"
    )

    model_idle_seconds: int = Field(default=3600, ge=1)
    kv_cache_idle_seconds: int = Field(default=3600, ge=1)


def requirement_key(fields: dict) -> tuple:
    return (fields.get("model"), "context" in fields, fields.get("context"))


def gib(budget: list[int]) -> str:
    return ", ".join(f"{value / GIB:.1f}" for value in budget)


def settings(models: ModelStoreStatus) -> tuple:
    """What a model store snapshot says that can change a memory requirement."""
    return (
        models.selected_tier,
        models.selected_whisper,
        models.whisper_formatting,
        tuple(
            sorted(
                (model.id, model.status, model.context_limit) for model in models.models
            )
        ),
    )


def worker_token() -> str:
    return os.getenv("HIKARI_WORKER_TOKEN") or "insecure-api-key"


def authorized(websocket: WebSocket) -> bool:
    token = worker_token()
    supplied = websocket.headers.get("authorization", "")
    return bool(token) and hmac.compare_digest(
        supplied.encode(), f"Bearer {token}".encode()
    )


def target_worker(request: InferenceTask) -> str | None:
    """The worker type for a job; None for store work (settings, downloads)."""
    if request.operation in ("download", "configure"):
        return None
    model = requested_model(request) or EXPERIMENTAL_FEATURES.get(request.feature)
    experiment = experimental_worker(model)
    if experiment is not None:
        if request.operation in ("load", "unload"):
            raise InferenceFailure(
                "Experimental models load per request; manual residency is unavailable.",
                422,
            )
        return experiment
    return FEATURE_WORKERS[request.feature]


def requested_model(request: InferenceTask) -> str | None:
    for field in ("model_id", "model"):
        value = getattr(request.input, field, None)
        if isinstance(value, str):
            return value
    return None


class Computer:
    """One physical computer: its RAM and GPUs, shared by its worker containers."""

    def __init__(self, info: dict):
        self.id: str = info["id"]
        self.name: str = info["name"]
        # Free memory when its first worker connects, before anything loads.
        self.ledger = MachineLedger(capacity=[info["ram_free"]], workers={})
        self.gpus: dict[str, int] = {}

    def gpu_index(self, device: dict) -> int:
        index = self.gpus.get(device["uuid"])
        if index is None:
            index = self.ledger.add_gpu(device["vram_free"])
            self.gpus[device["uuid"]] = index
        return index


class Worker:
    """One worker container's connection."""

    def __init__(self, websocket: WebSocket, hello: dict, computer: Computer):
        self.id = uuid4().hex
        self.websocket = websocket
        self.kind: str = hello["worker"]
        if self.kind not in PRIORITY:
            raise ValueError(f"Unknown worker type {self.kind!r}")
        self.instance: str = hello["instance"]
        self.name = f"{self.kind}@{self.instance}"
        self.computer = computer
        # The worker's own GPU order mapped onto its computer's ledger.
        self.devices = [
            computer.gpu_index(device) for device in hello["machine"]["devices"]
        ]
        self.models: ModelStoreStatus | None = None
        # A budget grant permits work; only engine observations prove residency.
        self.resident_model: str | None = None
        self.resident_context: dict[str, int | None] = {}
        self.residency_known = True
        self.ready = asyncio.Event()
        self.controls: dict[str, asyncio.Future] = {}
        self.preparing: asyncio.Task | None = None
        self.jobs: set[str] = set()
        self.write_lock = asyncio.Lock()

    @property
    def slot(self) -> WorkerSlot:
        return self.computer.ledger.workers[self.id]

    def local(self, budget: list[int]) -> list[int]:
        """A ledger budget in this worker's own [host, GPU...] order."""
        return [budget[0], *(budget[index] for index in self.devices)]

    def ledger(self, local: list[int]) -> list[int]:
        """This worker's own [host, GPU...] budget in ledger order."""
        budget = [0] * len(self.computer.ledger.capacity)
        budget[0] = local[0]
        for index, value in zip(self.devices, local[1:]):
            budget[index] = value
        return budget

    async def send(self, message: dict | bytes, *, job: InferenceJob | None = None):
        text = message if isinstance(message, bytes) else encode_request(message)
        async with self.write_lock:
            if job is not None:
                if job.cancel_requested or job.failure is not None:
                    return
                if self.id not in self.computer.ledger.workers:
                    raise InferenceFailure("Worker disconnected.", 503)
                job.machines.add(self.id)
                self.jobs.add(job.request.id)
                if job.worker is not None:
                    self.slot.pending += 1
                job.sent = True
            # Once entered, a write may have delivered even if its caller leaves.
            write = asyncio.create_task(
                asyncio.wait_for(self.websocket.send_text(text.decode()), 10)
            )
            try:
                await asyncio.shield(write)
            except asyncio.CancelledError:
                if job is not None:
                    job.request_cancel()
                with suppress(Exception):
                    await await_cleanup(write)
                raise

    async def control(self, command: str, timeout: float = 30, **fields) -> dict:
        request_id = uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self.controls[request_id] = future
        try:
            await self.send({"id": request_id, "command": command, **fields})
            return await asyncio.wait_for(future, timeout)
        finally:
            self.controls.pop(request_id, None)

    def free_bytes(self) -> int:
        free = self.computer.ledger.free()
        return free[0] + max((free[index] for index in self.devices), default=0)


class WorkerHub:
    """The API's view of every worker container."""

    def __init__(self, on_event=None):
        self.on_event = on_event
        worker_settings = WorkerSettings()
        self.model_idle_seconds = worker_settings.model_idle_seconds
        self.kv_cache_idle_seconds = worker_settings.kv_cache_idle_seconds
        self.workers: dict[str, Worker] = {}
        self.computers: dict[str, Computer] = {}
        self.jobs: dict[str, InferenceJob] = {}
        self.memory: WorkerMemory | None = None
        self.memory_observed_at: str | None = None
        self.primary: str | None = None
        # The first worker's model store; settings are mirrored to the rest.
        self.models: ModelStoreStatus | None = None
        self.wake = asyncio.Event()
        self.joined = asyncio.Condition()
        self.scheduler: asyncio.Task | None = None
        self.closing = False
        self.calls: dict[str, frozenset[str]] = {}

    def _refresh_models(self):
        """Append opt-in checkpoints without replacing the native tier or defaults."""
        primary = self.workers.get(self.primary or "")
        base = primary.models if primary is not None else None
        snapshot = (
            base.model_copy(deep=True)
            if base is not None
            else ModelStoreStatus(models=[])
        )
        known = {model.id for model in snapshot.models}
        for worker in self.workers.values():
            if not worker.kind.startswith("lfm3_") or worker.models is None:
                continue
            for model in worker.models.models:
                try:
                    kind = experimental_worker(model.id)
                except InferenceFailure:
                    continue
                if kind == worker.kind and model.id not in known:
                    snapshot.models.append(model.model_copy(deep=True))
                    known.add(model.id)
        self.models = snapshot if base is not None or snapshot.models else None

    @contextmanager
    def full_duplex(self, models: frozenset[str]):
        """Keep call participants at equal priority until their last call ends.

        This changes retention, not grants: normal physical admission and release
        acknowledgement still apply when the three models cannot fit together.
        """
        call_id = uuid4().hex
        self.calls[call_id] = models
        self._refresh_full_duplex()
        self.wake.set()
        try:
            yield
        finally:
            self.calls.pop(call_id, None)
            self._refresh_full_duplex()
            self.wake.set()

    def _refresh_full_duplex(self):
        active = set().union(*self.calls.values())
        for computer in self.computers.values():
            for slot in computer.ledger.workers.values():
                participating = (
                    slot.kind in ("llm", "tts", "speech")
                    and slot.requirement is not None
                    and slot.requirement.model in active
                )
                if slot.full_duplex and not participating:
                    # Give a released participant its normal idle period.
                    slot.last_used = time.monotonic()
                slot.full_duplex = participating

    async def start(self):
        self.scheduler = asyncio.create_task(self._schedule_loop())

    async def close(self):
        self.closing = True
        if self.scheduler is not None:
            self.scheduler.cancel()
            with suppress(asyncio.CancelledError):
                await self.scheduler
        for worker in tuple(self.workers.values()):
            with suppress(Exception):
                await worker.websocket.close()

    # Connections

    async def serve(self, websocket: WebSocket):
        """Registers one worker container and relays its messages until it leaves."""
        try:
            hello = json.loads(await asyncio.wait_for(websocket.receive_text(), 30))
            if hello.get("state") != "hello" or hello.get("protocol") != PROTOCOL:
                raise ValueError("Unsupported worker protocol")
            info = hello["machine"]
            computer = self.computers.get(info["id"]) or Computer(info)
            worker = Worker(websocket, hello, computer)
            for model, (kind, _) in EXPERIMENTAL_MODELS.items():
                if worker.kind == kind:
                    experimental_worker(model)
        except (
            TimeoutError,
            ValueError,
            KeyError,
            TypeError,
            InferenceFailure,
        ) as error:
            log.warning("Rejected worker: %s", error)
            await websocket.close(code=1008)
            return
        self.computers[computer.id] = computer
        computer.ledger.workers[worker.id] = WorkerSlot(
            name=worker.id,
            kind=worker.kind,
            devices=worker.devices,
            budget=[0] * len(computer.ledger.capacity),
            # A reconnect has no completion history; allow a full idle period.
            last_used=time.monotonic(),
        )
        self.workers[worker.id] = worker
        if self.primary is None and not worker.kind.startswith("lfm3_"):
            self.primary = worker.id
        log.info("Worker %s connected on %s", worker.name, computer.name)
        try:
            while True:
                text = await websocket.receive_text()
                try:
                    await self._dispatch(worker, json.loads(text))
                except (ValueError, ValidationError, KeyError) as error:
                    log.warning("Invalid message from %s: %s", worker.name, error)
        except WebSocketDisconnect:
            pass
        finally:
            await self._disconnected(worker)

    async def _dispatch(self, worker: Worker, message: dict):
        state = message.get("state")
        if state == "models":
            models = ModelStoreStatus.model_validate(message["models"])
            # Every finished job resends the store; only a change to settings
            # or checkpoints can change what a worker needs.
            if worker.models is None or settings(models) != settings(worker.models):
                worker.slot.requirement = None
                self.wake.set()
            worker.models = models
            self._refresh_models()
            return
        if state == "ready":
            # Routable only once its settings match the primary's.
            worker.preparing = asyncio.create_task(self._prepare(worker))
            return
        if state == "worker":
            if message.get("worker") == worker.kind:
                slot = worker.slot
                slot.loaded = message["loaded"]
                slot.budget = worker.ledger(list(message["budget"]))
                slot.releasing = False
                if "model" in message:
                    if worker.resident_model != message["model"]:
                        worker.resident_context = {}
                    worker.resident_model = message["model"]
                    worker.residency_known = True
                log.info(
                    "%s %s; budget %s GiB",
                    worker.name,
                    message.get("event", "updated"),
                    gib(slot.budget),
                )
            self.wake.set()
            return
        future = worker.controls.get(message.get("id", ""))
        if future is not None:
            if future.done():
                return
            if state == "rejected":
                future.set_exception(
                    InferenceFailure(
                        message.get("error") or "Worker command rejected.",
                        message.get("status_code", 502),
                    )
                )
            else:
                future.set_result(message)
            return
        # Tracing metadata is optional and cannot reject a valid job event.
        spans = message.pop("trace_spans", None)
        job = self.jobs.get(message.get("id", ""))
        if (
            state in TERMINAL
            and job is not None
            and worker.id in job.machines
            and job.tracing is not None
            and isinstance(spans, list)
        ):
            for native in spans[:16]:
                job.tracing.native(worker, native)
        self._job_event(worker, WorkerEvent.model_validate(message))

    async def _disconnected(self, worker: Worker):
        self.workers.pop(worker.id, None)
        computer = worker.computer
        computer.ledger.workers.pop(worker.id, None)
        if not computer.ledger.workers:
            self.computers.pop(computer.id, None)
        if self.primary == worker.id:
            self.primary = next(
                (
                    current.id
                    for current in self.workers.values()
                    if not current.kind.startswith("lfm3_")
                ),
                None,
            )
        self._refresh_models()
        for future in worker.controls.values():
            if not future.done():
                future.set_exception(InferenceFailure("Worker disconnected.", 503))
        log.warning("Worker %s disconnected", worker.name)
        for job_id in tuple(worker.jobs):
            job = self.jobs.get(job_id)
            if job is None:
                continue
            job.machines.discard(worker.id)
            if job.cancel_requested or job.failure is not None:
                if not job.machines:
                    self._settle(job)
                continue
            if job.worker is not None and job.state in REQUEUEABLE:
                try:
                    await self._send_job(job)
                    continue
                except InferenceFailure:
                    pass
            failure = InferenceFailure("Worker disconnected.", 503)
            if job.worker is None:
                job.failure = job.failure or failure
                if not job.machines:
                    self._settle(job)
            else:
                self._fail(job, failure)
        self.wake.set()

    # Jobs

    def _ready(self, kind: str | None = None) -> list[Worker]:
        return [
            worker
            for worker in self.workers.values()
            if worker.ready.is_set() and (kind is None or worker.kind == kind)
        ]

    def _route(self, kind: str) -> Worker:
        candidates = self._ready(kind)
        if not candidates:
            raise InferenceFailure(f"No {kind} worker is connected.", 503)
        loaded = [worker for worker in candidates if worker.slot.loaded]
        return max(loaded or candidates, key=Worker.free_bytes)

    def _store_targets(self, request: InferenceTask) -> list[Worker]:
        """Settings go to every worker; a download to one worker of its type per computer."""
        if request.operation == "configure":
            model = requested_model(request)
            kind = experimental_worker(model)
            targets = self._ready(kind) if kind else self._ready()
        else:
            kind = experimental_worker(request.input.model_id)
            for model in self.models.models if self.models is not None else []:
                if model.id == request.input.model_id and kind is None:
                    kind = CHECKPOINT_WORKERS.get(model.kind)
            targets, computers = [], set()
            for worker in self._ready(kind):
                if kind is None and worker.kind.startswith("lfm3_"):
                    continue
                if worker.computer.id not in computers:
                    computers.add(worker.computer.id)
                    targets.append(worker)
        if not targets:
            raise InferenceFailure("No worker is connected for this model.", 503)
        return targets

    async def _send_job(self, job: InferenceJob):
        if job.result.done():
            return
        if job.cancel_requested or job.failure is not None:
            self._settle(job)
            return
        message = job.request.model_dump(mode="json")
        if job.tracing is not None:
            message.update(job.tracing.parent)
        targets = (
            self._store_targets(job.request)
            if job.worker is None
            else [self._route(job.worker)]
        )
        # A terminal event can arrive during send, including an overlapping requeue.
        job.submissions += 1
        try:
            for worker in targets:
                if job.cancel_requested or job.failure is not None:
                    break
                if worker.id not in self.workers:
                    job.failure = InferenceFailure("Worker disconnected.", 503)
                    break
                await worker.send(message, job=job)
        except BaseException as error:
            if isinstance(error, InferenceFailure):
                job.failure = job.failure or error
            job.request_cancel()
            raise
        finally:
            job.submissions -= 1
            self._settle(job)
            self.wake.set()

    async def _wait_for_worker(self, request: InferenceTask):
        """Gives worker containers that are still starting time to connect."""
        kind = target_worker(request)
        deadline = time.monotonic() + CONNECT_TIMEOUT
        async with self.joined:
            while not self._ready(kind):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                with suppress(TimeoutError):
                    await asyncio.wait_for(self.joined.wait(), remaining)

    async def submit(self, request: InferenceTask, *, streaming=False) -> InferenceJob:
        if self.closing:
            raise InferenceFailure("Inference is stopping.")
        await self._wait_for_worker(request)
        # Reject oversized input before registering a job that was never queued.
        encode_request(request.model_dump(mode="json"))
        if request.id in self.jobs:
            raise InferenceFailure("Request is already active.", 409)
        job = InferenceJob(request, streaming)
        job.tracing = JobTracing()
        job.worker = target_worker(request)
        self.jobs[request.id] = job
        try:
            await self._send_job(job)
            await asyncio.wait_for(asyncio.shield(job.accepted), 10)
        except BaseException:
            if job.sent:
                with suppress(Exception, asyncio.CancelledError):
                    await await_cleanup(asyncio.create_task(self.cancel_and_join(job)))
            for future in (job.accepted, job.result):
                if future.done() and not future.cancelled():
                    future.exception()
            self._release(job)
            self.jobs.pop(request.id, None)
            job.tracing.end(failed=True)
            raise
        return job

    async def offload(self, kind: str):
        """Releases every worker of a type, e.g. after an unload."""
        for worker in tuple(self.workers.values()):
            slot = worker.slot
            if worker.kind != kind or not slot.loaded or slot.releasing:
                continue
            await self._apply(worker.computer, [Action("offload", worker.id)])

    async def cancel(self, job_id: str):
        job = self.jobs.get(job_id)
        if job is None or job.result.done():
            return
        job.request_cancel()
        if not any(worker_id in self.workers for worker_id in job.machines):
            job.machines.clear()
            self._fail(job, InferenceFailure("Inference cancelled.", 499))
            return
        send_error = None
        for worker_id in tuple(job.machines):
            worker = self.workers.get(worker_id)
            if worker is not None:
                try:
                    await worker.send({"id": job_id, "command": "cancel"})
                except (
                    InferenceFailure,
                    RuntimeError,
                    OSError,
                    TimeoutError,
                    WebSocketDisconnect,
                ) as error:
                    send_error = send_error or error
        if send_error is not None:
            raise send_error

    async def cancel_and_join(self, job: InferenceJob):
        job.request_cancel()
        # A job whose workers are all gone has no one left to report its end.
        if not job.result.done() and not any(
            worker_id in self.workers for worker_id in job.machines
        ):
            job.machines.clear()
            self._fail(job, InferenceFailure("Inference cancelled.", 499))
        if not job.result.done():
            try:
                await self.cancel(job.request.id)
            except (
                InferenceFailure,
                RuntimeError,
                OSError,
                TimeoutError,
                WebSocketDisconnect,
            ):
                # A failed send does not confirm cleanup; wait for an end or disconnect.
                pass
        with suppress(Exception):
            await asyncio.shield(job.result)

    def _release(self, job: InferenceJob, worker_id: str | None = None):
        for current in [worker_id] if worker_id else tuple(job.machines):
            worker = self.workers.get(current)
            if worker is None or job.request.id not in worker.jobs:
                continue
            worker.jobs.discard(job.request.id)
            if job.worker is not None:
                slot = worker.slot
                slot.pending = max(0, slot.pending - 1)
                slot.busy = False
                slot.last_used = time.monotonic()
        self.wake.set()

    def _fail(self, job: InferenceJob, failure: InferenceFailure):
        job.failure = job.failure or failure
        failure = job.failure
        if job.submissions:
            return
        if job.tracing is not None:
            job.tracing.end(failed=True)
        job.identified.set()
        if not job.accepted.done():
            job.accepted.set_exception(failure)
        if not job.result.done():
            job.result.set_exception(failure)
        self._release(job)
        job.machines.clear()

    def _settle(self, job: InferenceJob):
        if job.submissions or job.machines or job.result.done():
            return
        if job.tracing is not None:
            job.tracing.end(failed=job.failure is not None)
        job.identified.set()
        if not job.accepted.done():
            if job.failure is not None:
                job.accepted.set_exception(job.failure)
            else:
                job.accepted.set_result(None)
        if job.failure is not None:
            job.result.set_exception(job.failure)
        else:
            job.result.set_result(job.outcome)

    def _job_event(self, worker: Worker, event: WorkerEvent):
        job = self.jobs.get(event.id)
        if job is None or worker.id not in job.machines:
            log.debug("Event for unknown job %s from %s", event.id, worker.name)
            return
        if job.tracing is not None:
            job.tracing.event(worker, event)
        if event.memory is not None and worker.kind == "llm":
            self.memory = event.memory
            self.memory_observed_at = datetime.now(UTC).isoformat()
        if event.model is not None:
            job.model = event.model
            job.identified.set()
        if job.request.feature == "llm" and event.context is not None:
            job.context = event.context
        if (
            worker.kind == "llm"
            and job.request.feature == "llm"
            and event.state == "succeeded"
        ):
            worker.resident_model = (
                None if job.request.operation == "unload" else job.model
            )
            worker.resident_context = (
                job.context if worker.resident_model is not None else {}
            )
            worker.residency_known = (
                job.request.operation == "unload" or job.model is not None
            )
        elif (
            worker.kind == "llm"
            and job.request.feature == "llm"
            and event.state in ("failed", "cancelled")
        ):
            # Preparation cancellation can leave an old engine resident, while
            # execution failure can destroy it without a control snapshot.
            worker.residency_known = False
            worker.resident_model = None
            worker.resident_context = {}
        if event.state == "audio_transcript":
            if (
                job.request.feature != "audio_call"
                or job.request.input.mode != "turn"
                or job.events is None
            ):
                job.stream_error = "Unexpected audio transcript event."
                asyncio.create_task(self.cancel(event.id))
            elif job.stream_error is None:
                try:
                    job.events.put_nowait(
                        {"type": "transcript", "text": event.transcript}
                    )
                except asyncio.QueueFull:
                    job.stream_error = "Audio turn consumer is too slow."
                    asyncio.create_task(self.cancel(event.id))
            return
        if event.state == "content":
            if job.events is not None and job.stream_error is None:
                try:
                    job.events.put_nowait({"content": event.content})
                except asyncio.QueueFull:
                    asyncio.create_task(self.cancel(event.id))
                    job.stream_error = "Chat stream consumer is too slow."
            return
        if event.state == "progress":
            if self.on_event is not None:
                self.on_event(job, event)
            return
        if event.state == "waiting_for_resources" and job.state != JobState.WAITING:
            # The grant may predate a change in what the worker needs.
            if job.worker is not None:
                worker.slot.requirement = None
                self.wake.set()
            log.warning(
                "%s job %s waits for memory: %s",
                worker.name,
                job.request.id,
                event.error or "no reason given",
            )
        job.state = JobState(event.state)
        if job.worker is not None:
            worker.slot.busy = event.state in ("preparing", "loading", "running")
        # Acceptance is the first queue; downloads may take far longer to finish.
        if event.state == "queued" and not job.accepted.done():
            job.accepted.set_result(None)
        if self.on_event is not None:
            self.on_event(job, event)
        if event.state not in TERMINAL:
            return
        job.machines.discard(worker.id)
        self._release(job, worker.id)
        if event.state == "succeeded":
            if job.outcome is None or worker.id == self.primary:
                job.outcome = event.result
        elif job.failure is None:
            status = (
                (429 if event.error == "queue_full" else 503)
                if event.state == "rejected"
                else 499
                if event.state == "cancelled"
                else event.status_code
            )
            job.failure = InferenceFailure(
                event.error or "Inference cancelled.", status
            )
        if not job.machines:
            self._settle(job)

    # Settings consistency across workers

    async def _synchronize(self, worker: Worker):
        """Bring a newly connected worker's settings in line with the primary."""
        primary = self.workers.get(self.primary or "")
        if primary is None or primary is worker:
            return
        source, target = primary.models, worker.models
        if source is None or target is None:
            return
        tasks = []
        if source.selected_tier and source.selected_tier != target.selected_tier:
            tasks.append({"feature": "tier_selection", "tier": source.selected_tier})
        if (
            source.selected_whisper
            and source.selected_whisper != target.selected_whisper
        ):
            tasks.append(
                {"feature": "whisper_selection", "model": source.selected_whisper}
            )
        if source.whisper_formatting != target.whisper_formatting:
            tasks.append(
                {"feature": "whisper_formatting", "enabled": source.whisper_formatting}
            )
        limits = {model.id: model.context_limit for model in target.models}
        for model in source.models:
            if (
                model.kind == "llm"
                and model.id in limits
                and limits[model.id] != model.context_limit
            ):
                tasks.append(
                    {
                        "feature": "model_context",
                        "model_id": model.id,
                        "context_limit": model.context_limit,
                    }
                )
        for task in tasks:
            job = InferenceJob(InferenceTask(operation="configure", input=task), False)
            job.machines.add(worker.id)
            self.jobs[job.request.id] = job
            worker.jobs.add(job.request.id)
            try:
                await worker.send(job.request.model_dump(mode="json"))
                await asyncio.wait_for(asyncio.shield(job.result), 60)
            finally:
                worker.jobs.discard(job.request.id)
                self.jobs.pop(job.request.id, None)

    async def _prepare(self, worker: Worker):
        """Synchronizes settings, then makes the worker routable."""
        try:
            await self._synchronize(worker)
        except Exception:
            log.exception("Could not synchronize settings to %s", worker.name)
            await worker.websocket.close(code=1011)
            return
        worker.ready.set()
        async with self.joined:
            self.joined.notify_all()
        self.wake.set()

    # Scheduling

    async def _schedule_loop(self):
        while True:
            with suppress(TimeoutError):
                await asyncio.wait_for(self.wake.wait(), 1)
            self.wake.clear()
            for computer in tuple(self.computers.values()):
                try:
                    await self._schedule(computer)
                except InferenceFailure as error:
                    log.warning("Scheduling on %s failed: %s", computer.name, error)
                except Exception:
                    log.exception("Scheduling on %s failed", computer.name)

    def _head_fields(self, worker: Worker) -> dict:
        """What the oldest queued job asks for that changes the worker's needs."""
        for job in self.jobs.values():
            if job.worker != worker.kind or worker.id not in job.machines:
                continue
            fields = {"model": requested_model(job.request)}
            # A load's explicit context overrides the saved one, null included.
            if (
                job.request.operation == "load"
                and "context_limit" in job.request.input.model_fields_set
            ):
                fields["context"] = job.request.input.context_limit
            return fields
        return {"model": None}

    async def _requirement(self, worker: Worker) -> Requirement:
        fields = self._head_fields(worker)
        try:
            reply = await worker.control("requirement", **fields)
            requirement = Requirement.parse(reply)
            requirement.key = requirement_key(fields)
            return requirement
        except InferenceFailure as error:
            # Preparation reports the real problem (no tier, no download) once
            # the worker runs; a host-only budget lets it get that far.
            log.info("No memory requirement for %s: %s", worker.name, error)
            return Requirement(
                model="", minimum=(0, 0), preferred=(0, 0), key=requirement_key(fields)
            )

    async def _apply(self, computer: Computer, actions: list[Action]) -> bool:
        for action in actions:
            worker = self.workers[action.worker]
            slot = worker.slot
            release = releases(slot, action)
            if release:
                # Mark before sending: the worker may confirm the release before
                # its reply to this command arrives. The old charge stays until
                # its "worker" event updates the ledger and wakes planning. If
                # the command fails, the timeout rereads the worker's state.
                slot.releasing = True
                slot.releasing_since = time.monotonic()
            if action.kind == "offload":
                reply = await worker.control("offload")
            else:
                reply = await worker.control(
                    "load",
                    budget=worker.local(action.budget),
                    kv_cache_idle_seconds=self.kv_cache_idle_seconds,
                )
            if not reply.get("ok"):
                # The worker started a job in the meantime; nothing changed.
                slot.releasing = False
                log.info("%s %s refused: busy", action.kind, worker.name)
                return False
            if release:
                log.info("%s %s requested", action.kind, worker.name)
                return False
            computer.ledger.apply(action)
            log.info("%s %s with %s GiB", action.kind, worker.name, gib(action.budget))
        return True

    async def _reject_unfittable(self, worker: Worker, requirement: Requirement):
        """Fails jobs that need more memory than the worker's computer has."""
        host, device = (value / GIB for value in requirement.minimum)
        capacity = worker.computer.ledger.capacity
        failure = InferenceFailure(
            f"{requirement.model or worker.kind} needs at least {host:.1f} GiB RAM and "
            f"{device:.1f} GiB of VRAM; {worker.name} has "
            f"{capacity[0] / GIB:.1f} GiB RAM and "
            f"{sum(capacity[index] for index in worker.devices) / GIB:.1f} GiB free "
            f"across {len(worker.devices)} GPU(s).",
            507,
        )
        log.warning("%s", failure)
        for job_id in tuple(worker.jobs):
            job = self.jobs.get(job_id)
            if job is not None and job.worker == worker.kind:
                with suppress(InferenceFailure, RuntimeError, OSError, TimeoutError):
                    await worker.send({"id": job_id, "command": "cancel"})
                self._fail(job, failure)

    async def _resynchronize(self, computer: Computer):
        """Rereads workers whose release was never confirmed.

        A worker still cleaning up keeps its old charge. Otherwise it is charged
        the larger of its budget and what its engine still holds, so memory is
        never credited while allocations remain.
        """
        for slot in computer.ledger.workers.values():
            worker = self.workers.get(slot.name)
            if worker is None or not slot.releasing:
                continue
            reply = await worker.control("status")
            for status in reply["workers"]:
                if status["worker"] != worker.kind:
                    continue
                if status.get("releasing"):
                    slot.releasing_since = time.monotonic()
                    continue
                local = [
                    max(budget, used)
                    for budget, used in zip(status["budget"], status.get("used", []))
                ] or list(status["budget"])
                slot.loaded = status["loaded"]
                if "model" in status:
                    if worker.resident_model != status["model"]:
                        worker.resident_context = {}
                    worker.resident_model = status["model"]
                    worker.residency_known = True
                slot.budget = worker.ledger(local)
                slot.releasing = False

    async def _schedule(self, computer: Computer):
        ledger = computer.ledger
        # Nothing is planned on top of memory that is still being released.
        releasing = [slot for slot in ledger.workers.values() if slot.releasing]
        if releasing:
            if any(
                time.monotonic() - slot.releasing_since > RELEASE_TIMEOUT
                for slot in releasing
            ):
                log.warning(
                    "Release on %s was not confirmed; rereading budgets", computer.name
                )
                await self._resynchronize(computer)
            return
        workers = {
            slot.name: self.workers[slot.name]
            for slot in ledger.workers.values()
            if slot.name in self.workers and self.workers[slot.name].ready.is_set()
        }
        slots = [worker.slot for worker in workers.values()]
        # A new head job may ask for another model or context.
        for slot in slots:
            if (
                slot.pending
                and slot.requirement is not None
                and slot.requirement.key is not None
                and slot.requirement.key
                != requirement_key(self._head_fields(workers[slot.name]))
            ):
                slot.requirement = None
        # Resident workers keep current requirements so they can shrink and grow.
        for slot in slots:
            if slot.requirement is None and (slot.loaded or slot.pending):
                slot.requirement = await self._requirement(workers[slot.name])
        self._refresh_full_duplex()
        waiting = sorted(
            (slot for slot in slots if needs_grant(ledger, slot)),
            key=lambda slot: (slot.priority, slot.last_used),
        )
        for slot in waiting:
            # The unloaded language tier keeps room for its minimum.
            for other in slots:
                if (
                    other.requirement is None
                    and not other.loaded
                    and other.priority == TOP_PRIORITY < slot.priority
                ):
                    other.requirement = await self._requirement(workers[other.name])
            if slot.requirement is None:
                slot.requirement = await self._requirement(workers[slot.name])
            if ledger.fit(slot.requirement, ledger.capacity, slot.devices) is None:
                await self._reject_unfittable(workers[slot.name], slot.requirement)
                continue
            actions = plan_load(ledger, slot.name)
            if actions is None:
                continue
            if not await self._apply(computer, actions):
                return
        # Requests take precedence, including queued jobs that already have a
        # budget. Recheck after the requirement and grant RPCs above.
        if not any(slot.pending for slot in ledger.workers.values()):
            now = time.monotonic()
            for slot in slots:
                if (
                    slot.loaded
                    and not slot.busy
                    and not slot.releasing
                    and not slot.full_duplex
                    and now - slot.last_used >= self.model_idle_seconds
                ):
                    await self._apply(computer, [Action("offload", slot.name)])
                    return
        await self._apply(computer, plan_growth(ledger))
