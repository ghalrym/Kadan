"""Durable ownership of logical agents, independent of workload concurrency."""

import asyncio
import logging
from datetime import UTC, datetime
from uuid import uuid4

from ..api.config import ServiceError
from ..api.pydantic_models.agent_runs import AgentRun
from ..api.pydantic_models.runners import Runner
from ..api.services import LocalServices
from .agent_runs import AgentRuns
from .config import RunnerSettings
from .image_jobs import ImageJobs
from .pet_jobs import PetJobs

logger = logging.getLogger(__name__)
POLL_SECONDS = 2


class RunnerServices(LocalServices):
    def __init__(self, client, settings, owner):
        super().__init__(client, settings)
        self.runs = AgentRuns(self, owner)
        self.images = ImageJobs(self)
        self.pets = PetJobs(self)


class AgentRunner:
    def __init__(self, services: RunnerServices, settings: RunnerSettings):
        self.services = services
        self.settings = settings
        self.owner = services.runs.owner
        self.work: dict[str, tuple[AgentRun, asyncio.Task]] = {}

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
        values = await self.services.json_request("Tsuchi", "GET", "/api/agent-runs")
        runs = [AgentRun.model_validate(value) for value in values]
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
    services.images.start()
    services.pets.start()
    try:
        await runner.run(stop)
    finally:
        await services.images.close()
        await services.pets.close()
