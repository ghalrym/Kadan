"""Durable ownership of logical agents, independent of workload concurrency."""

import asyncio
import logging
import os
from datetime import UTC, datetime
from uuid import uuid4

from ..api.config import ServiceError
from ..api.pydantic_models.agent_runs import AgentRun
from ..api.pydantic_models.runners import Runner
from ..api.services import LocalServices
from .agent_runs import AgentRuns
from .compaction import IdleCompactor
from .config import RunnerSettings
from .group_chat_events import GroupChatWebhookEvents
from .image_jobs import ImageJobs
from .memory_queue import MemoryQueue
from .pet_jobs import PetJobs
from .t3_events import T3ThreadEvents
from ..work.runner import WorkRunner

logger = logging.getLogger(__name__)
POLL_SECONDS = 0.5


class RunnerServices(LocalServices):
    def __init__(self, client, settings, owner):
        super().__init__(client, settings)
        self.runs = AgentRuns(self, owner)
        self.images = ImageJobs(self)
        self.pets = PetJobs(self)
        self.t3 = T3ThreadEvents(self)
        self.group_chat_events = GroupChatWebhookEvents(self)
        self.work_runner = WorkRunner(self)
        self.compactor = IdleCompactor(self)
        self.memory_queue = MemoryQueue(self)


class AgentRunner:
    def __init__(self, services: RunnerServices, settings: RunnerSettings):
        self.services = services
        self.settings = settings
        self.owner = services.runs.owner
        self.work: dict[str, tuple[AgentRun, asyncio.Task]] = {}
        self.assigned_keys: tuple[str, ...] = ()

    async def assignments(self) -> Runner:
        path = f"/api/runners/{self.settings.id}"
        # This renews only infrastructure ownership. It never wakes an agent.
        await self.services.json_request(
            "Tsuchi",
            "PUT",
            path,
            json={"worker_id": str(self.owner), "capacity": self.settings.capacity},
        )
        runner = Runner.model_validate(
            await self.services.json_request(
                "Tsuchi", "POST", path + "/assign", json={"worker_id": str(self.owner)}
            )
        )
        if (
            runner.runner_id != self.settings.id
            or runner.worker_id != self.owner
            or runner.capacity != self.settings.capacity
            or runner.lease_expires_at <= datetime.now(UTC)
        ):
            raise ServiceError(
                "Runner ownership changed.", 409, "runner_ownership_lost"
            )
        return runner

    async def reconcile(self) -> None:
        runner = await self.assignments()
        self.assigned_keys = tuple(runner.agent_keys)
        values = await self.services.json_request("Tsuchi", "GET", "/api/agent-runs")
        runs = [AgentRun.model_validate(value) for value in values]
        if self.work or any(run.status in {"queued", "running"} for run in runs):
            memory_queue = getattr(self.services, "memory_queue", None)
            if memory_queue is not None:
                memory_queue.preempt()
        by_id = {run.id: run for run in runs}
        for key, (run, task) in list(self.work.items()):
            current = by_id.get(run.id)
            if (
                key not in runner.agent_keys
                or current is None
                or current.status in {"completed", "failed", "cancelled"}
                or (current.status == "running" and current.owner_id != self.owner)
            ):
                task.cancel()
            if task.done() or task.cancelling():
                result = await asyncio.gather(task, return_exceptions=True)
                if isinstance(result[0], Exception):
                    logger.error("Agent execution ended: %s", result[0])
                del self.work[key]
        # One resumable graph task per assigned logical agent. Idle assignments
        # stay in Tsuchi and consume capacity without allocating a task/thread.
        for run in sorted(runs, key=lambda item: (item.created_at, str(item.id))):
            if run.agent_key not in runner.agent_keys or run.agent_key in self.work:
                continue
            if run.status == "queued" or (
                run.status == "running" and run.lease_expires_at <= datetime.now(UTC)
            ):
                task = asyncio.create_task(
                    self.services.runs.process(run), name=f"agent-{run.agent_key}"
                )
                self.work[run.agent_key] = (run, task)

    async def close(self) -> None:
        for _, task in self.work.values():
            task.cancel()
        await asyncio.gather(
            *(task for _, task in self.work.values()), return_exceptions=True
        )
        self.work.clear()

    async def run(self, stop: asyncio.Event) -> None:
        try:
            while not stop.is_set():
                try:
                    await self.reconcile()
                except ServiceError as error:
                    await self.close()
                    if error.status == 409:
                        # A fenced process must exit, never take its session back.
                        raise
                    logger.exception("Runner could not reconcile durable work")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=POLL_SECONDS)
                except TimeoutError:
                    pass
        finally:
            await self.close()


async def serve(client, settings, runner_settings: RunnerSettings, stop: asyncio.Event):
    services = RunnerServices(client, settings, uuid4())
    runner = AgentRunner(services, runner_settings)
    services.compactor.set_agent_keys_provider(lambda: runner.assigned_keys)
    services.memory_queue.set_agent_keys_provider(lambda: runner.assigned_keys)
    services.images.start()
    services.pets.start()
    services.t3.start()
    services.group_chat_events.start()
    run_server_work_runner = os.environ.get(
        "KADAN_SERVER_WORK_RUNNER", ""
    ).strip().lower() in {"1", "true", "yes"}
    if run_server_work_runner:
        services.work_runner.start()
    services.compactor.start()
    services.memory_queue.start()
    try:
        await runner.run(stop)
    finally:
        await services.memory_queue.close()
        await services.compactor.close()
        if run_server_work_runner:
            await services.work_runner.close()
        await services.images.close()
        await services.pets.close()
        await services.group_chat_events.close()
        await services.t3.close()
