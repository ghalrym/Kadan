"""Runner scheduling uses durable assignments without executing idle agents."""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from kadan.api.config import ServiceError
from kadan.api.pydantic_models.agent_runs import AgentRun
from kadan.api.pydantic_models.runners import Runner
from kadan.api.run_client import RunClient
from kadan.runner.config import RunnerSettings
from kadan.runner.service import AgentRunner


def make_run(key="agent", **changes):
    now = datetime.now(UTC)
    return AgentRun(
        id=uuid4(),
        agent_key=key,
        conversation_id=uuid4(),
        version=1,
        created_at=now,
        updated_at=now,
        **changes,
    )


class Boundary:
    def __init__(self, keys=(), runs=()):
        self.owner = uuid4()
        self.keys = list(keys)
        self.values = list(runs)
        self.started = []
        self.cancelled = []
        self.release = asyncio.Event()
        self.error = None
        self.calls = []
        self.runs = SimpleNamespace(owner=self.owner, process=self.process)

    async def process(self, run):
        self.started.append(run.id)
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.append(run.id)
            raise

    async def json_request(self, service, method, path, **kwargs):
        self.calls.append((service, method, path, kwargs))
        if self.error:
            raise self.error
        if path == "/api/agent-runs":
            return [run.model_dump(mode="json") for run in self.values]
        return Runner(
            runner_id="runner-1",
            worker_id=self.owner,
            capacity=5,
            lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
            agent_keys=self.keys,
        ).model_dump(mode="json")


def test_capacity_default_and_environment_override(monkeypatch):
    monkeypatch.delenv("KADAN_RUNNER_CAPACITY", raising=False)
    assert RunnerSettings(_env_file=None).capacity == 5
    monkeypatch.setenv("KADAN_RUNNER_CAPACITY", "8")
    assert RunnerSettings(_env_file=None).capacity == 8


@pytest.mark.parametrize(
    "capacity", [True, False, 1.5, 0, -1, "1.5", "true", "0", "-1"]
)
def test_capacity_rejects_invalid_values(capacity):
    with pytest.raises(ValidationError):
        RunnerSettings(capacity=capacity, _env_file=None)


def test_idle_assignments_consume_slots_without_execution():
    async def scenario():
        boundary = Boundary([f"agent-{index}" for index in range(5)])
        runner = AgentRunner(boundary, RunnerSettings(_env_file=None))
        await runner.reconcile()
        assert runner.work == {}
        assert boundary.started == []
        assert boundary.calls[0][3]["json"]["capacity"] == 5

    asyncio.run(scenario())


def test_assigned_keys_run_concurrently_with_one_task_each():
    async def scenario():
        first, second, other, excess = (
            make_run("a"),
            make_run("a"),
            make_run("b"),
            make_run("excess"),
        )
        boundary = Boundary(["a", "b"], [first, second, other, excess])
        runner = AgentRunner(boundary, RunnerSettings(_env_file=None))
        try:
            await runner.reconcile()
            await asyncio.sleep(0)
            assert set(boundary.started) == {first.id, other.id}
            await runner.reconcile()
            await asyncio.sleep(0)
            assert len(boundary.started) == 2
        finally:
            await runner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("reason", ["cancelled", "reassigned", "foreign_owner"])
def test_lost_or_cancelled_assignment_stops_execution(reason):
    async def scenario():
        run = make_run()
        boundary = Boundary(["agent"], [run])
        runner = AgentRunner(boundary, RunnerSettings(_env_file=None))
        await runner.reconcile()
        await asyncio.sleep(0)
        if reason == "reassigned":
            boundary.keys = []
        elif reason == "foreign_owner":
            boundary.values = [
                run.model_copy(
                    update={
                        "status": "running",
                        "owner_id": uuid4(),
                        "lease_expires_at": datetime.now(UTC) + timedelta(seconds=60),
                    }
                )
            ]
        else:
            boundary.values = [run.model_copy(update={"status": "cancelled"})]
        await runner.reconcile()
        assert boundary.cancelled == [run.id]
        assert runner.work == {}

    asyncio.run(scenario())


def test_completed_task_releases_key_for_next_durable_run():
    async def scenario():
        first, next_run = make_run(), make_run()
        boundary = Boundary(["agent"], [first, next_run])
        runner = AgentRunner(boundary, RunnerSettings(_env_file=None))
        await runner.reconcile()
        await asyncio.sleep(0)
        boundary.values = [first.model_copy(update={"status": "completed"}), next_run]
        boundary.release.set()
        await asyncio.sleep(0)
        await runner.reconcile()
        await asyncio.sleep(0)
        assert boundary.started == [first.id, next_run.id]
        await runner.close()

    asyncio.run(scenario())


def test_duplicate_runner_ownership_exits_and_cancels_work():
    async def scenario():
        run = make_run()
        boundary = Boundary(["agent"], [run])
        runner = AgentRunner(boundary, RunnerSettings(_env_file=None))
        await runner.reconcile()
        await asyncio.sleep(0)
        boundary.error = ServiceError("Ownership lost", 409)
        with pytest.raises(ServiceError):
            await runner.run(asyncio.Event())
        assert boundary.cancelled == [run.id]
        assert runner.work == {}

    asyncio.run(scenario())


def test_transport_failure_cancels_work_before_retry(monkeypatch):
    async def scenario():
        run = make_run()
        boundary = Boundary(["agent"], [run])
        runner = AgentRunner(boundary, RunnerSettings(_env_file=None))
        await runner.reconcile()
        await asyncio.sleep(0)
        boundary.error = ServiceError("Unavailable", 503)
        stop = asyncio.Event()
        original_request = boundary.json_request
        retries = []

        async def request_after_failure(service, method, path, **kwargs):
            if boundary.error:
                error, boundary.error = boundary.error, None
                raise error
            if method == "PUT":
                assert boundary.cancelled == [run.id]
                assert runner.work == {}
                retries.append(path)
                stop.set()
            return await original_request(service, method, path, **kwargs)

        monkeypatch.setattr(boundary, "json_request", request_after_failure)
        monkeypatch.setattr("kadan.runner.service.POLL_SECONDS", 0.001)
        await runner.run(stop)
        assert retries == ["/api/runners/runner-1"]
        assert boundary.cancelled == [run.id]
        assert runner.work == {}

    asyncio.run(scenario())


def test_api_run_client_only_persists_enqueue_and_cancellation():
    async def scenario():
        run = make_run()
        calls = []

        async def request(*args, **kwargs):
            calls.append((args, kwargs))
            return run.model_dump(mode="json")

        client = RunClient(SimpleNamespace(json_request=request))
        value = SimpleNamespace(
            message_id=uuid4(),
            content="Hello",
            persona_id=None,
            expected_version=1,
        )
        assert await client.enqueue("agent", run.conversation_id, value) == run
        assert await client.cancel("agent", run.id) == run
        assert [call[0][1] for call in calls] == ["POST", "POST"]
        assert calls[0][1]["json"]["message"]["content"] == "Hello"
        assert calls[1][0][2].endswith(f"/{run.id}/cancel")
        assert not hasattr(client, "wakeup")
        assert not hasattr(client, "process")

    asyncio.run(scenario())


def test_recovery_starts_only_expired_runs():
    async def scenario():
        expired = make_run(
            "expired",
            status="running",
            owner_id=uuid4(),
            lease_expires_at=datetime.now(UTC) - timedelta(seconds=1),
        )
        live = make_run(
            "live",
            status="running",
            owner_id=uuid4(),
            lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
        )
        boundary = Boundary(["expired", "live"], [expired, live])
        runner = AgentRunner(boundary, RunnerSettings(_env_file=None))
        try:
            await runner.reconcile()
            await asyncio.sleep(0)
            assert boundary.started == [expired.id]
        finally:
            await runner.close()

    asyncio.run(scenario())
