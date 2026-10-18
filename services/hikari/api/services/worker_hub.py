"""Connected worker machines, job routing, and memory scheduling across them.

Each machine runs one native process with a worker per inference type. The hub
owns every memory decision: workers only pull jobs while the hub has granted
them a budget.
"""

import asyncio
import hmac
import json
import logging
import os
import time
from contextlib import suppress
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
    LocalMachine,
    WorkerEvent,
    WorkerMemory,
    encode_request,
)
from api.services.memory_scheduler import (
    GIB,
    TOP_PRIORITY,
    Action,
    MachineLedger,
    Requirement,
    needs_grant,
    plan_growth,
    plan_load,
    releases,
)
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

log = logging.getLogger(__name__)

PROTOCOL = 3

# Features served by an inference worker; the rest run on every machine's store.
FEATURE_WORKERS = {
    "llm": "llm",
    "decisions": "decision",
    "tts": "tts",
    "stt": "speech",
    "stt_formatting": "formatting",
    "image": "image",
    "video": "video",
}

# Jobs that have not started executing can move to another machine.
REQUEUEABLE = (JobState.QUEUED, JobState.PREPARING, JobState.WAITING)

# A release the worker never confirms is reread from its status after this.
RELEASE_TIMEOUT = 60


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
    return os.getenv("HIKARI_WORKER_TOKEN", "")


def authorized(websocket: WebSocket) -> bool:
    token = worker_token()
    supplied = websocket.headers.get("authorization", "")
    return bool(token) and hmac.compare_digest(
        supplied.encode(), f"Bearer {token}".encode()
    )


def target_worker(request: InferenceTask) -> str | None:
    """The inference worker for a job; None for store work on every machine."""
    if request.operation in ("download", "configure"):
        return None
    return FEATURE_WORKERS[request.feature]


def requested_model(request: InferenceTask) -> str | None:
    for field in ("model_id", "model"):
        value = getattr(request.input, field, None)
        if isinstance(value, str):
            return value
    return None


class Machine:
    def __init__(self, hub: "WorkerHub", websocket: WebSocket, hello: dict):
        self.hub = hub
        self.websocket = websocket
        self.name: str = hello["machine"]["name"]
        self.info: dict = hello["machine"]
        self.workers: list[str] = hello["workers"]
        self.ledger = MachineLedger.from_machine(self.info, self.workers)
        self.models: ModelStoreStatus | None = None
        self.ready = asyncio.Event()
        self.controls: dict[str, asyncio.Future] = {}
        self.preparing: asyncio.Task | None = None
        self.jobs: set[str] = set()
        self.write_lock = asyncio.Lock()

    async def send(self, message: dict | bytes):
        text = message if isinstance(message, bytes) else encode_request(message)
        async with self.write_lock:
            await asyncio.wait_for(self.websocket.send_text(text.decode()), 10)

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
        free = self.ledger.free()
        return free[0] + max(free[1:], default=0)


class WorkerHub:
    """The API's view of every worker; replaces a single local FIFO."""

    def __init__(self, on_event=None, local: LocalMachine | None = None):
        self.on_event = on_event
        self.local = local if local is not None else LocalMachine()
        self.machines: dict[str, Machine] = {}
        self.jobs: dict[str, InferenceJob] = {}
        self.memory: WorkerMemory | None = None
        self.memory_observed_at: str | None = None
        self.primary: str | None = None
        # The primary machine's model store; settings are mirrored to the rest.
        self.models: ModelStoreStatus | None = None
        self.wake = asyncio.Event()
        self.connected = asyncio.Event()
        self.scheduler: asyncio.Task | None = None
        self.closing = False

    @property
    def last_exit(self):
        return self.local.last_exit

    async def start(self):
        self.scheduler = asyncio.create_task(self._schedule_loop())
        if os.getenv("HIKARI_LOCAL_WORKER", "1") == "0":
            return
        token = worker_token()
        if not token:
            raise InferenceFailure("HIKARI_WORKER_TOKEN is not configured.")
        url = os.getenv("HIKARI_WORKER_API_URL", "ws://127.0.0.1:8000/v1/workers")
        await self.local.start(url, token)

    async def wait_connected(self, timeout: float):
        try:
            await asyncio.wait_for(self.connected.wait(), timeout)
        except TimeoutError:
            exit = self.local.last_exit
            raise InferenceFailure(
                exit.describe() if exit else "No inference worker machine connected."
            ) from None

    async def close(self):
        self.closing = True
        if self.scheduler is not None:
            self.scheduler.cancel()
            with suppress(asyncio.CancelledError):
                await self.scheduler
        for machine in tuple(self.machines.values()):
            with suppress(Exception):
                await machine.websocket.close()
        await self.local.close()

    # Connections

    async def serve(self, websocket: WebSocket):
        """Registers one machine and relays its messages until it disconnects."""
        try:
            hello = json.loads(await asyncio.wait_for(websocket.receive_text(), 30))
            if hello.get("state") != "hello" or hello.get("protocol") != PROTOCOL:
                raise ValueError("Unsupported worker protocol")
            machine = Machine(self, websocket, hello)
        except (TimeoutError, ValueError, KeyError, TypeError) as error:
            log.warning("Rejected worker machine: %s", error)
            await websocket.close(code=1008)
            return
        if machine.name in self.machines:
            log.warning("Rejected duplicate worker machine %s", machine.name)
            await websocket.close(code=1008)
            return
        self.machines[machine.name] = machine
        if self.primary is None:
            self.primary = machine.name
        log.info("Worker machine %s connected", machine.name)
        try:
            while True:
                text = await websocket.receive_text()
                try:
                    await self._dispatch(machine, json.loads(text))
                except (ValueError, ValidationError, KeyError) as error:
                    log.warning("Invalid message from %s: %s", machine.name, error)
        except WebSocketDisconnect:
            pass
        finally:
            await self._disconnected(machine)

    async def _dispatch(self, machine: Machine, message: dict):
        state = message.get("state")
        if state == "models":
            models = ModelStoreStatus.model_validate(message["models"])
            # Every finished job resends the store; only a change to settings
            # or checkpoints can change what a worker needs.
            if machine.models is None or settings(models) != settings(machine.models):
                for slot in machine.ledger.workers.values():
                    slot.requirement = None
                self.wake.set()
            machine.models = models
            if machine.name == self.primary:
                self.models = machine.models
            return
        if state == "ready":
            # Routable only once its settings match the primary's.
            machine.preparing = asyncio.create_task(self._prepare(machine))
            return
        if state == "worker":
            slot = machine.ledger.workers.get(message["worker"])
            if slot is not None:
                slot.loaded = message["loaded"]
                slot.budget = list(message["budget"])
                slot.releasing = False
                log.info(
                    "%s %s on %s; budget %s GiB",
                    message["worker"],
                    message.get("event", "updated"),
                    machine.name,
                    gib(slot.budget),
                )
            self.wake.set()
            return
        future = machine.controls.get(message.get("id", ""))
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
        self._job_event(machine, WorkerEvent.model_validate(message))

    async def _disconnected(self, machine: Machine):
        self.machines.pop(machine.name, None)
        if self.primary == machine.name:
            self.primary = next(iter(self.machines), None)
            primary = self.machines.get(self.primary or "")
            self.models = primary.models if primary is not None else None
        if not self.machines:
            self.connected.clear()
        for future in machine.controls.values():
            if not future.done():
                future.set_exception(
                    InferenceFailure("Worker machine disconnected.", 503)
                )
        log.warning("Worker machine %s disconnected", machine.name)
        for job_id in tuple(machine.jobs):
            job = self.jobs.get(job_id)
            if job is None:
                continue
            job.machines.discard(machine.name)
            if job.worker is not None and job.state in REQUEUEABLE:
                try:
                    await self._send_job(job)
                    continue
                except InferenceFailure:
                    pass
            failure = InferenceFailure("Worker machine disconnected.", 503)
            if job.worker is None:
                job.failure = job.failure or failure
                if not job.machines:
                    self._settle(job)
            else:
                self._fail(job, failure)
        self.wake.set()

    # Jobs

    def _route(self, worker: str) -> Machine:
        candidates = [
            machine
            for machine in self.machines.values()
            if machine.ready.is_set() and worker in machine.workers
        ]
        if not candidates:
            raise InferenceFailure(f"No connected machine runs {worker} work.", 503)
        loaded = [
            machine for machine in candidates if machine.ledger.workers[worker].loaded
        ]
        return max(loaded or candidates, key=Machine.free_bytes)

    async def _send_job(self, job: InferenceJob):
        message = job.request.model_dump(mode="json")
        if job.worker is None:
            targets = [m for m in self.machines.values() if m.ready.is_set()]
            if not targets:
                raise InferenceFailure("No inference worker machine connected.", 503)
        else:
            targets = [self._route(job.worker)]
        for machine in targets:
            job.machines.add(machine.name)
            machine.jobs.add(job.request.id)
            if job.worker is not None:
                machine.ledger.workers[job.worker].pending += 1
            await machine.send(message)
        job.sent = True
        self.wake.set()

    async def submit(self, request: InferenceTask, *, streaming=False) -> InferenceJob:
        if self.closing:
            raise InferenceFailure("Inference worker is stopping.")
        # The local machine connects only after the API starts serving.
        if not self.connected.is_set():
            await self.wait_connected(60)
        # Reject oversized input before registering a job that was never queued.
        encode_request(request.model_dump(mode="json"))
        if request.id in self.jobs:
            raise InferenceFailure("Request is already active.", 409)
        job = InferenceJob(request, streaming)
        job.worker = target_worker(request)
        self.jobs[request.id] = job
        try:
            await self._send_job(job)
            await asyncio.wait_for(asyncio.shield(job.accepted), 10)
        except BaseException:
            if job.sent:
                with suppress(Exception, asyncio.CancelledError):
                    await await_cleanup(asyncio.create_task(self.cancel_and_join(job)))
            self._release(job)
            self.jobs.pop(request.id, None)
            raise
        return job

    async def offload(self, worker: str):
        """Releases a worker's memory on every machine, e.g. after an unload."""
        for machine in tuple(self.machines.values()):
            slot = machine.ledger.workers.get(worker)
            if slot is None or not slot.loaded or slot.releasing:
                continue
            action = Action("offload", worker)
            await self._apply(machine, [action])

    async def cancel(self, job_id: str):
        job = self.jobs.get(job_id)
        if job is None:
            return
        for name in tuple(job.machines):
            machine = self.machines.get(name)
            if machine is not None:
                await machine.send({"id": job_id, "command": "cancel"})

    async def cancel_and_join(self, job: InferenceJob):
        # A job whose machines are all gone has no one left to report its end.
        if not job.result.done() and not any(
            name in self.machines for name in job.machines
        ):
            self._fail(job, InferenceFailure("Inference cancelled.", 499))
        if not job.result.done():
            try:
                await self.cancel(job.request.id)
            except (InferenceFailure, RuntimeError, OSError, TimeoutError):
                # The machine is unreachable; it no longer runs this job.
                self._fail(job, InferenceFailure("Inference cancelled.", 499))
        with suppress(Exception):
            await asyncio.shield(job.result)

    def _release(self, job: InferenceJob, machine_name: str | None = None):
        for name in [machine_name] if machine_name else tuple(job.machines):
            machine = self.machines.get(name)
            if machine is None or job.request.id not in machine.jobs:
                continue
            machine.jobs.discard(job.request.id)
            if job.worker is not None:
                slot = machine.ledger.workers[job.worker]
                slot.pending = max(0, slot.pending - 1)
                slot.busy = False
                slot.last_used = time.monotonic()
        self.wake.set()

    def _fail(self, job: InferenceJob, failure: InferenceFailure):
        job.identified.set()
        if not job.accepted.done():
            job.accepted.set_exception(failure)
        if not job.result.done():
            job.result.set_exception(failure)
        self._release(job)
        job.machines.clear()

    def _settle(self, job: InferenceJob):
        job.identified.set()
        if not job.accepted.done():
            if job.failure is not None:
                job.accepted.set_exception(job.failure)
            else:
                job.accepted.set_result(None)
        if job.result.done():
            return
        if job.failure is not None:
            job.result.set_exception(job.failure)
        else:
            job.result.set_result(job.outcome)

    def _job_event(self, machine: Machine, event: WorkerEvent):
        job = self.jobs.get(event.id)
        if job is None or machine.name not in job.machines:
            log.debug("Event for unknown job %s from %s", event.id, machine.name)
            return
        if event.memory is not None and event.worker == "llm":
            self.memory = event.memory
            self.memory_observed_at = datetime.now(UTC).isoformat()
        if event.model is not None:
            job.model = event.model
            job.identified.set()
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
                machine.ledger.workers[job.worker].requirement = None
                self.wake.set()
            log.warning(
                "%s job %s waits for memory on %s: %s",
                job.worker,
                job.request.id,
                machine.name,
                event.error or "no reason given",
            )
        job.state = JobState(event.state)
        if job.worker is not None:
            slot = machine.ledger.workers[job.worker]
            slot.busy = event.state in ("preparing", "loading", "running")
        # Acceptance is the first queue; downloads may take far longer to finish.
        if event.state == "queued" and not job.accepted.done():
            job.accepted.set_result(None)
        if self.on_event is not None:
            self.on_event(job, event)
        if event.state not in TERMINAL:
            return
        job.machines.discard(machine.name)
        self._release(job, machine.name)
        if event.state == "succeeded":
            if job.outcome is None or machine.name == self.primary:
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

    # Settings consistency across machines

    async def _synchronize(self, machine: Machine):
        """Bring a newly connected machine's settings in line with the primary."""
        primary = self.machines.get(self.primary or "")
        if primary is None or primary is machine:
            return
        source, target = primary.models, machine.models
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
            job.machines.add(machine.name)
            self.jobs[job.request.id] = job
            machine.jobs.add(job.request.id)
            try:
                await machine.send(job.request.model_dump(mode="json"))
                await asyncio.wait_for(asyncio.shield(job.result), 60)
            finally:
                machine.jobs.discard(job.request.id)
                self.jobs.pop(job.request.id, None)

    async def _prepare(self, machine: Machine):
        """Synchronizes settings, then makes the machine routable."""
        try:
            await self._synchronize(machine)
        except Exception:
            log.exception("Could not synchronize settings to %s", machine.name)
            await machine.websocket.close(code=1011)
            return
        machine.ready.set()
        self.connected.set()
        self.wake.set()

    # Scheduling

    async def _schedule_loop(self):
        while True:
            with suppress(TimeoutError):
                await asyncio.wait_for(self.wake.wait(), 1)
            self.wake.clear()
            for machine in tuple(self.machines.values()):
                if not machine.ready.is_set():
                    continue
                try:
                    await self._schedule(machine)
                except InferenceFailure as error:
                    log.warning("Scheduling on %s failed: %s", machine.name, error)
                except Exception:
                    log.exception("Scheduling on %s failed", machine.name)

    def _head_fields(self, machine: Machine, worker: str) -> dict:
        """What the oldest queued job asks for that changes the worker's needs."""
        for job in self.jobs.values():
            if job.worker != worker or machine.name not in job.machines:
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

    async def _requirement(self, machine: Machine, worker: str) -> Requirement:
        fields = self._head_fields(machine, worker)
        try:
            reply = await machine.control("requirement", worker=worker, **fields)
            requirement = Requirement.parse(reply)
            requirement.key = requirement_key(fields)
            return requirement
        except InferenceFailure as error:
            # Preparation reports the real problem (no tier, no download) once
            # the worker runs; a host-only budget lets it get that far.
            log.info(
                "No memory requirement for %s on %s: %s", worker, machine.name, error
            )
            return Requirement(
                model="", minimum=(0, 0), preferred=(0, 0), key=requirement_key(fields)
            )

    async def _apply(self, machine: Machine, actions: list[Action]) -> bool:
        for action in actions:
            slot = machine.ledger.workers[action.worker]
            release = releases(slot, action)
            if release:
                # Mark before sending: the worker may confirm the release before
                # its reply to this command arrives. The old charge stays until
                # its "worker" event updates the ledger and wakes planning. If
                # the command fails, the timeout rereads the worker's state.
                slot.releasing = True
                slot.releasing_since = time.monotonic()
            if action.kind == "offload":
                reply = await machine.control("offload", worker=action.worker)
            else:
                reply = await machine.control(
                    "load", worker=action.worker, budget=action.budget
                )
            if not reply.get("ok"):
                # The worker started a job in the meantime; nothing changed.
                slot.releasing = False
                log.info(
                    "%s %s on %s refused: busy",
                    action.kind,
                    action.worker,
                    machine.name,
                )
                return False
            if release:
                log.info(
                    "%s %s on %s requested", action.kind, action.worker, machine.name
                )
                return False
            machine.ledger.apply(action)
            log.info(
                "%s %s on %s with %s GiB",
                action.kind,
                action.worker,
                machine.name,
                gib(action.budget),
            )
        return True

    async def _reject_unfittable(
        self, machine: Machine, worker: str, requirement: Requirement
    ):
        """Fails jobs that need more memory than the whole machine has."""
        host, device = (value / GIB for value in requirement.minimum)
        devices = len(machine.ledger.capacity) - 1
        capacity = machine.ledger.capacity
        failure = InferenceFailure(
            f"{requirement.model or worker} needs at least {host:.1f} GiB RAM and "
            f"{device:.1f} GiB of VRAM; {machine.name} has "
            f"{capacity[0] / GIB:.1f} GiB RAM and "
            f"{sum(capacity[1:]) / GIB:.1f} GiB free across {devices} GPU(s).",
            507,
        )
        log.warning("%s", failure)
        for job_id in tuple(machine.jobs):
            job = self.jobs.get(job_id)
            if job is not None and job.worker == worker:
                with suppress(InferenceFailure, RuntimeError, OSError, TimeoutError):
                    await machine.send({"id": job_id, "command": "cancel"})
                self._fail(job, failure)

    async def _resynchronize(self, machine: Machine):
        """Rereads workers whose release was never confirmed.

        A worker still cleaning up keeps its old charge. Otherwise it is charged
        the larger of its budget and what its engine still holds, so memory is
        never credited while allocations remain.
        """
        reply = await machine.control("status")
        for worker in reply["workers"]:
            slot = machine.ledger.workers.get(worker["worker"])
            if slot is None or not slot.releasing:
                continue
            if worker.get("releasing"):
                slot.releasing_since = time.monotonic()
                continue
            slot.loaded = worker["loaded"]
            slot.budget = [
                max(budget, used)
                for budget, used in zip(worker["budget"], worker.get("used", []))
            ] or list(worker["budget"])
            slot.releasing = False

    async def _schedule(self, machine: Machine):
        ledger = machine.ledger
        # Nothing is planned on top of memory that is still being released.
        releasing = [slot for slot in ledger.workers.values() if slot.releasing]
        if releasing:
            if any(
                time.monotonic() - slot.releasing_since > RELEASE_TIMEOUT
                for slot in releasing
            ):
                log.warning(
                    "Release on %s was not confirmed; rereading budgets", machine.name
                )
                await self._resynchronize(machine)
            return
        # A new head job may ask for another model or context.
        for slot in ledger.workers.values():
            if (
                slot.pending
                and slot.requirement is not None
                and slot.requirement.key is not None
                and slot.requirement.key
                != requirement_key(self._head_fields(machine, slot.name))
            ):
                slot.requirement = None
        # Resident workers keep current requirements so they can shrink and grow.
        for slot in ledger.workers.values():
            if slot.requirement is None and (slot.loaded or slot.pending):
                slot.requirement = await self._requirement(machine, slot.name)
        waiting = sorted(
            (slot for slot in ledger.workers.values() if needs_grant(ledger, slot)),
            key=lambda slot: (slot.priority, slot.last_used),
        )
        for slot in waiting:
            # The unloaded language tier keeps room for its minimum.
            for other in ledger.workers.values():
                if (
                    other.requirement is None
                    and not other.loaded
                    and other.priority == TOP_PRIORITY < slot.priority
                ):
                    other.requirement = await self._requirement(machine, other.name)
            if slot.requirement is None:
                slot.requirement = await self._requirement(machine, slot.name)
            if ledger.fit(slot.requirement, ledger.capacity) is None:
                await self._reject_unfittable(machine, slot.name, slot.requirement)
                continue
            actions = plan_load(ledger, slot.name)
            if actions is None:
                continue
            if not await self._apply(machine, actions):
                return
        await self._apply(machine, plan_growth(ledger))
