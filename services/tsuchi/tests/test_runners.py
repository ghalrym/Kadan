from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError
from tsuchi.config import StorageError
from tsuchi.models import (
    AgentInput,
    AgentRunState,
    ConversationContent,
    Message,
    RunMessage,
    utc_now,
)
from tsuchi.repository import Repository


@pytest.fixture
def repository(tmp_path):
    repository = Repository(tmp_path)
    repository.initialize()
    for index in range(8):
        repository.create_agent(AgentInput(key=f"agent-{index}", name=f"Agent {index}"))
    return repository


def test_idle_assignment_persists_and_excess_waits(repository):
    owner = uuid4()
    assert repository.register_runner("one", owner).capacity == 5
    first = repository.assign_runner("one", owner)
    assert first.agent_keys == [f"agent-{i}" for i in range(5)]
    assert repository.assign_runner("one", owner).agent_keys == first.agent_keys
    other = uuid4()
    repository.register_runner("two", other, 2)
    assert repository.assign_runner("two", other).agent_keys == ["agent-5", "agent-6"]
    assert (
        len({key for runner in repository.list_runners() for key in runner.agent_keys})
        == 7
    )


def test_competing_assignments_are_atomic(repository):
    workers = [uuid4(), uuid4()]
    for index, worker in enumerate(workers):
        repository.register_runner(f"runner-{index}", worker)
    with ThreadPoolExecutor(2) as pool:
        runners = list(
            pool.map(
                lambda pair: repository.assign_runner(f"runner-{pair[0]}", pair[1]),
                enumerate(workers),
            )
        )
    assert sum(len(r.agent_keys) for r in runners) == 8
    assert not set(runners[0].agent_keys) & set(runners[1].agent_keys)


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5, "5"])
def test_capacity_requires_positive_integer(repository, capacity):
    with pytest.raises(ValidationError):
        repository.register_runner("one", uuid4(), capacity)


def test_active_session_rejects_duplicate_and_expired_session_recovers(repository):
    original, replacement = uuid4(), uuid4()
    repository.register_runner("one", original)
    assigned = repository.assign_runner("one", original).agent_keys
    with pytest.raises(StorageError, match="replacement"):
        repository.register_runner("one", replacement)
    with repository.connect(write=True) as db:
        db.execute(
            "UPDATE runners SET lease_expires_at = ?",
            ((utc_now() - timedelta(seconds=1)).isoformat(),),
        )
    with pytest.raises(StorageError, match="ownership"):
        repository.assign_runner("one", original)
    recovered = repository.register_runner("one", replacement)
    assert recovered.agent_keys == assigned
    with repository.connect() as db:
        with pytest.raises(StorageError, match="not owned"):
            repository._assigned_worker(db, assigned[0], original)
        repository._assigned_worker(db, assigned[0], replacement)


def test_explicit_reassignment_checks_capacity_and_expected_assignment(repository):
    first, second = uuid4(), uuid4()
    repository.register_runner("one", first, 1)
    repository.register_runner("two", second, 1)
    repository.assign_runner("one", first)
    repository.assign_runner("two", second)
    with pytest.raises(StorageError, match="capacity"):
        repository.reassign_agent("agent-0", "two", "one")
    with pytest.raises(StorageError, match="changed"):
        repository.reassign_agent("agent-0", None, None)
    repository.reassign_agent("agent-1", None, "two")
    repository.reassign_agent("agent-0", "two", "one")
    assert repository.list_runners()[1].agent_keys == ["agent-0"]
    with pytest.raises(ValidationError):
        repository.register_runner("two", second, 0)


@pytest.mark.parametrize("change", ["replace", "reassign"])
def test_ownership_change_expires_runs_preserving_transcript(repository, change):
    owner = uuid4()
    repository.register_runner("one", owner)
    repository.assign_runner("one", owner)
    resource = repository.create_resource(
        "agent-0", ConversationContent(kind="conversation", title="Conversation")
    )
    run = repository.create_run(
        "agent-0",
        resource.id,
        Message(id=uuid4(), role="user", content="Hello", created_at=utc_now()),
        resource.version,
    )
    transcript = [RunMessage(role="assistant", content="Persisted action")]
    state = AgentRunState(
        status="running",
        owner_id=owner,
        lease_expires_at=utc_now() + timedelta(seconds=60),
        transcript=transcript,
    )
    run = repository.update_run("agent-0", run.id, state, run.version, owner)
    replacement = uuid4()
    if change == "replace":
        repository.register_runner("one", replacement, expected_worker_id=owner)
    else:
        repository.register_runner("two", replacement)
        repository.reassign_agent("agent-0", "two", "one")
    recovered = repository.get_run("agent-0", run.id)
    assert recovered.transcript == transcript
    assert recovered.status == "running"
    assert recovered.version == run.version + 1
    assert recovered.lease_expires_at <= utc_now()
    with pytest.raises(StorageError, match="not owned"):
        repository.update_run("agent-0", run.id, state, recovered.version, owner)
    renewed = state.model_copy(
        update={
            "owner_id": replacement,
            "lease_expires_at": utc_now() + timedelta(seconds=60),
        }
    )
    assert (
        repository.update_run(
            "agent-0", run.id, renewed, recovered.version, replacement
        ).owner_id
        == replacement
    )
