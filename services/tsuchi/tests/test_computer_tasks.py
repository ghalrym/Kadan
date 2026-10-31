import json
from datetime import timedelta
from uuid import uuid4, uuid5

import pytest
from test_agent_runs import claim, enqueue
from tsuchi.config import StorageError
from tsuchi.models import AgentInput, ComputerThreadState, RunMessage, utc_now
from tsuchi.repository import Repository


@pytest.fixture
def repository(tmp_path):
    repository = Repository(tmp_path)
    repository.initialize()
    repository.create_agent(AgentInput(key="alpha", name="Alpha"))
    return repository


def handed_over(repository, instructions="Open the docs"):
    run = claim(repository, enqueue(repository))
    task = repository.create_computer_task(
        "alpha", uuid4(), instructions, run.id, run.owner_id
    )
    return run, task


def lease(owner, transcript=()):
    return ComputerThreadState(
        transcript=list(transcript),
        owner_id=owner,
        lease_expires_at=utc_now() + timedelta(seconds=60),
    )


def claim_thread(repository, owner):
    thread = repository.get_computer_thread("alpha")
    return repository.update_computer_thread(
        "alpha", lease(owner, thread.transcript), thread.version, owner, []
    )


def test_handover_is_idempotent_and_queues_the_task(repository):
    run, task = handed_over(repository)
    again = repository.create_computer_task(
        "alpha", task.id, "Open the docs", run.id, run.owner_id
    )
    assert again == task
    assert task.origin.conversation_id == run.conversation_id
    assert task.event_run_id == uuid5(task.id, "event")
    assert repository.active_computer_tasks() == [task]
    with pytest.raises(StorageError):
        repository.create_computer_task(
            "alpha", task.id, "Something else", run.id, run.owner_id
        )


def test_user_request_starts_a_task_when_idle_and_steers_when_busy(repository):
    kind, task = repository.computer_request("alpha", uuid4(), "Open a terminal")
    assert (kind, task.origin) == ("task", None)
    request = uuid4()
    kind, steered = repository.computer_request("alpha", request, "Use bash")
    assert (kind, steered.id) == ("steering", task.id)
    assert repository.computer_request("alpha", request, "Use bash")[0] == "steering"
    steering = repository.get_computer_thread("alpha").steering
    assert [item.content for item in steering] == ["Use bash"]


def test_finishing_wakes_the_origin_conversation_once(repository):
    run, task = handed_over(repository)
    owner = uuid4()
    thread = claim_thread(repository, owner)
    thread = repository.update_computer_thread(
        "alpha", lease(owner), thread.version, owner, [], task.id, "running"
    )
    finished = repository.update_computer_thread(
        "alpha",
        lease(owner, [RunMessage(role="user", content="done")]),
        thread.version,
        owner,
        [],
        task.id,
        "completed",
        "Docs are open.",
    )
    task = repository.get_computer_task("alpha", task.id)
    assert (task.status, task.summary) == ("completed", "Docs are open.")
    event = repository.get_run("alpha", task.event_run_id)
    assert event.conversation_id == run.conversation_id
    assert event.trigger_event.kind == "computer"
    assert json.loads(event.trigger_event.content)["summary"] == "Docs are open."
    assert finished.transcript[-1].content == "done"


def test_finishing_is_refused_while_steering_is_unread(repository):
    _, task = repository.computer_request("alpha", uuid4(), "Open a terminal")
    owner = uuid4()
    thread = claim_thread(repository, owner)
    thread = repository.update_computer_thread(
        "alpha", lease(owner), thread.version, owner, [], task.id, "running"
    )
    repository.computer_request("alpha", uuid4(), "Actually use the browser")
    with pytest.raises(StorageError, match="New steering arrived"):
        repository.update_computer_thread(
            "alpha", lease(owner), thread.version, owner, [], task.id, "completed"
        )
    steering = repository.get_computer_thread("alpha").steering
    thread = repository.update_computer_thread(
        "alpha",
        lease(owner),
        thread.version,
        owner,
        [item.id for item in steering],
        task.id,
        "completed",
    )
    assert thread.steering == []
    assert repository.get_computer_task("alpha", task.id).status == "completed"


def test_another_worker_cannot_take_a_live_thread(repository):
    owner = uuid4()
    thread = claim_thread(repository, owner)
    other = uuid4()
    with pytest.raises(StorageError, match="Another worker"):
        repository.update_computer_thread(
            "alpha", lease(other), thread.version, other, []
        )


def test_claiming_cannot_change_the_transcript(repository):
    thread = repository.get_computer_thread("alpha")
    owner = uuid4()
    with pytest.raises(StorageError, match="cannot change"):
        repository.update_computer_thread(
            "alpha",
            lease(owner, [RunMessage(role="user", content="x")]),
            thread.version,
            owner,
            [],
        )


def test_cancelled_task_ignores_a_late_result(repository):
    _, task = handed_over(repository)
    owner = uuid4()
    thread = claim_thread(repository, owner)
    thread = repository.update_computer_thread(
        "alpha", lease(owner), thread.version, owner, [], task.id, "running"
    )
    cancelled = repository.cancel_computer_task("alpha", task.id)
    assert repository.get_run("alpha", cancelled.event_run_id).trigger_event
    repository.update_computer_thread(
        "alpha", lease(owner), thread.version, owner, [], task.id, "completed"
    )
    assert repository.get_computer_task("alpha", task.id).status == "cancelled"
    assert repository.active_computer_tasks("alpha") == []


def test_latest_task_is_the_most_recent_request(repository):
    assert repository.latest_computer_task("alpha") is None
    repository.computer_request("alpha", uuid4(), "Open a terminal")
    _, task = handed_over(repository)
    assert repository.latest_computer_task("alpha").id == task.id


def test_cancelling_offline_retires_only_that_tasks_steering(repository):
    _, first = repository.computer_request("alpha", uuid4(), "First task")
    repository.computer_request("alpha", uuid4(), "First steering")
    repository.cancel_computer_task("alpha", first.id)
    _, second = repository.computer_request("alpha", uuid4(), "Second task")
    repository.computer_request("alpha", uuid4(), "Second steering")
    steering = repository.get_computer_thread("alpha").steering
    assert [(item.task_id, item.content) for item in steering] == [
        (second.id, "Second steering")
    ]
