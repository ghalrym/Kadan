from datetime import timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError
from test_agent_runs import (
    claim,
    claim_image,
    complete_image,
    enqueue,
    expire_run,
    image_job,
    message,
    run_state,
)
from tsuchi.config import StorageError
from tsuchi.models import AgentInput, AgentRunState, RunEvent, TaskCheckpoint, utc_now
from tsuchi.repository import Repository


@pytest.fixture
def repository(tmp_path):
    store = Repository(tmp_path)
    store.initialize()
    store.create_agent(AgentInput(key="alpha", name="Alpha"))
    store.create_agent(AgentInput(key="beta", name="Beta"))
    return store


def checkpoint(repository, run, task=None, **changes):
    value = TaskCheckpoint(
        action_id=uuid4(),
        task_id=task.id if task else None,
        expected_version=task.version if task else None,
        goal="Finish the report",
        next_step="Review the draft",
        **changes,
    )
    return repository.checkpoint_task("alpha", run.id, value, run.owner_id), value


def complete(repository, run):
    run = repository.get_run("alpha", run.id)
    return repository.update_run(
        "alpha",
        run.id,
        run_state(run, status="completed", owner_id=None, lease_expires_at=None),
        run.version,
        run.owner_id,
    )


def event(repository, run):
    return repository.create_event_run(
        "alpha",
        run.conversation_id,
        uuid4(),
        run.id,
        RunEvent(kind="codex", content="Finished the draft"),
    )


def test_checkpoint_survives_repository_restart(repository):
    run = claim(repository, enqueue(repository))
    task, _ = checkpoint(repository, run, progress="Draft saved")
    restarted = Repository(repository.data_dir)
    restarted.initialize()
    assert restarted.get_task("alpha", task.id).progress == "Draft saved"
    assert restarted.get_run("alpha", run.id).task_id == task.id


def test_atomic_question_and_retry_do_not_duplicate_message(repository):
    run = claim(repository, enqueue(repository))
    task, value = checkpoint(
        repository,
        run,
        status="waiting",
        wait_kind="user",
        blocker="Choose the format",
        message="Would you like a PDF?",
    )
    again = repository.checkpoint_task("alpha", run.id, value, run.owner_id)
    assert again == task
    resource = repository.get_resource("alpha", run.conversation_id)
    assert [m.content for m in resource.messages].count("Would you like a PDF?") == 1
    assert len(repository.get_run("alpha", run.id).delivered_message_ids) == 1


def test_receipt_replay_ignores_only_expected_version(repository):
    run = claim(repository, enqueue(repository))
    task, _ = checkpoint(repository, run)
    updated, value = checkpoint(repository, run, task, progress="Reviewed")
    retry = value.model_copy(update={"expected_version": updated.version})
    assert repository.checkpoint_task("alpha", run.id, retry, run.owner_id) == updated
    with pytest.raises(StorageError):
        repository.checkpoint_task(
            "alpha",
            run.id,
            retry.model_copy(update={"progress": "Changed payload"}),
            run.owner_id,
        )


def test_stale_owner_cannot_checkpoint(repository):
    run = claim(repository, enqueue(repository))
    _, value = checkpoint(repository, run)
    with pytest.raises(StorageError):
        repository.checkpoint_task("alpha", run.id, value, uuid4())


def test_question_requires_visible_message():
    with pytest.raises(ValidationError):
        TaskCheckpoint(
            action_id=uuid4(),
            goal="Report",
            next_step="Choose",
            status="waiting",
            wait_kind="user",
            blocker="Need a format",
        )


def test_only_one_unfinished_goal_per_conversation(repository):
    run = claim(repository, enqueue(repository))
    checkpoint(repository, run)
    with pytest.raises(StorageError):
        checkpoint(repository, run)


def test_wait_user_does_not_wake_on_timer_sweep(repository):
    run = claim(repository, enqueue(repository))
    task, _ = checkpoint(
        repository,
        run,
        status="waiting",
        wait_kind="user",
        blocker="Choose",
        message="Which format?",
    )
    complete(repository, run)
    assert repository.wake_tasks() == []
    assert repository.get_task("alpha", task.id).status == "waiting"


def test_queued_reply_before_question_checkpoint_keeps_task(repository):
    run = claim(repository, enqueue(repository))
    resource = repository.get_resource("alpha", run.conversation_id)
    reply = repository.create_run(
        "alpha", resource.id, message(content="PDF"), resource.version
    )
    task, _ = checkpoint(
        repository,
        run,
        status="waiting",
        wait_kind="user",
        blocker="Choose",
        message="Which format?",
    )
    complete(repository, run)
    reply = claim(repository, reply)
    assert reply.task_id == task.id
    assert repository.get_task("alpha", task.id).status == "active"


def test_event_before_wait_checkpoint_is_not_lost(repository):
    run = claim(repository, enqueue(repository))
    task, _ = checkpoint(repository, run)
    completion = event(repository, run)
    task, _ = checkpoint(repository, run, task, status="waiting", wait_kind="tool")
    complete(repository, run)
    claimed = claim(repository, completion)
    assert claimed.task_id == task.id
    assert repository.get_task("alpha", task.id).status == "active"


def test_timer_is_atomic_deduplicated_and_waits_for_current_run(
    repository, monkeypatch
):
    run = claim(repository, enqueue(repository))
    now = utc_now()
    task, _ = checkpoint(
        repository,
        run,
        status="waiting",
        wait_kind="timer",
        wake_at=now + timedelta(seconds=40),
    )
    monkeypatch.setattr(
        "tsuchi.repository.utc_now", lambda: now + timedelta(seconds=45)
    )
    assert repository.wake_tasks() == []
    complete(repository, run)
    assert len(repository.wake_tasks()) == 1
    assert repository.wake_tasks() == []
    assert len([r for r in repository.active_runs() if r.task_id == task.id]) == 1


def test_timer_rejects_immediate_spin(repository):
    run = claim(repository, enqueue(repository))
    with pytest.raises(StorageError):
        checkpoint(
            repository, run, status="waiting", wait_kind="timer", wake_at=utc_now()
        )


def test_pause_fences_checkpoints_and_suppresses_events(repository):
    run = claim(repository, enqueue(repository))
    task, value = checkpoint(repository, run)
    paused = repository.control_task("alpha", task.id, "pause", task.version)
    with pytest.raises(StorageError):
        repository.checkpoint_task("alpha", run.id, value, run.owner_id)
    assert event(repository, run).status == "cancelled"
    resumed = repository.control_task("alpha", task.id, "resume", paused.version)
    wake = repository.get_run("alpha", resumed.current_run_id)
    assert wake.status == "queued"
    assert "Finished the draft" in wake.trigger_event.content
    assert wake.transcript == []


def test_cancellation_never_revived_by_events(repository):
    run = claim(repository, enqueue(repository))
    task, _ = checkpoint(repository, run)
    repository.control_task("alpha", task.id, "cancel", task.version)
    assert event(repository, run).status == "cancelled"
    assert repository.get_task("alpha", task.id).status == "cancelled"
    assert repository.active_runs() == []


def test_response_without_checkpoint_cannot_claim_task_completed(repository):
    run = claim(repository, enqueue(repository))
    task, _ = checkpoint(repository, run)
    complete(repository, run)
    saved = repository.get_task("alpha", task.id)
    assert saved.status == "failed"
    assert "without a saved next step" in saved.blocker


def test_failed_response_fails_waiting_task(repository):
    run = claim(repository, enqueue(repository))
    task, _ = checkpoint(repository, run, status="waiting", wait_kind="tool")
    run = repository.get_run("alpha", run.id)
    repository.update_run(
        "alpha",
        run.id,
        AgentRunState(status="failed", error="Tool unavailable"),
        run.version,
        run.owner_id,
    )
    assert repository.get_task("alpha", task.id).blocker == "Tool unavailable"


def test_task_completion_detaches_queued_user_message(repository):
    run = claim(repository, enqueue(repository))
    task, _ = checkpoint(repository, run)
    resource = repository.get_resource("alpha", run.conversation_id)
    reply = repository.create_run(
        "alpha", resource.id, message(content="Thanks"), resource.version
    )
    task = repository.get_task("alpha", task.id)
    checkpoint(repository, run, task, status="completed")
    complete(repository, run)
    assert claim(repository, reply).task_id is None


def test_expired_run_recovery_preserves_saved_wait(repository):
    run = claim(repository, enqueue(repository))
    task, value = checkpoint(repository, run, status="waiting", wait_kind="tool")
    expired = expire_run(repository, repository.get_run("alpha", run.id))
    recovered = claim(repository, expired)
    replay = repository.checkpoint_task("alpha", run.id, value, recovered.owner_id)
    assert replay.status == "waiting"
    complete(repository, recovered)
    assert repository.get_task("alpha", task.id).status == "waiting"


def test_image_event_inherits_task_and_pause_retains_result(repository):
    run = claim(repository, enqueue(repository))
    task, _ = checkpoint(repository, run, status="waiting", wait_kind="tool")
    job = claim_image(repository, image_job(repository, run))
    task = repository.control_task("alpha", task.id, "pause", task.version)
    done = complete_image(repository, job)
    event_run = repository.get_run("alpha", done.event_run_id)
    assert event_run.status == "cancelled"
    assert event_run.task_id == task.id
    resumed = repository.control_task("alpha", task.id, "resume", task.version)
    assert (
        str(job.id)
        in repository.get_run("alpha", resumed.current_run_id).trigger_event.content
    )


def test_stop_response_pauses_and_settles_queued_events(repository):
    run = claim(repository, enqueue(repository))
    task, _ = checkpoint(repository, run)
    followup = event(repository, run)
    repository.cancel_run("alpha", run.id)
    assert repository.get_task("alpha", task.id).status == "paused"
    assert repository.get_run("alpha", followup.id).status == "cancelled"


def test_task_scope_and_stale_controls_conflict(repository):
    run = claim(repository, enqueue(repository))
    task, _ = checkpoint(repository, run)
    with pytest.raises(StorageError):
        repository.get_task("beta", task.id)
    with pytest.raises(StorageError):
        repository.control_task("alpha", task.id, "pause", task.version + 1)
    assert repository.get_task("alpha", task.id).status == "active"


def test_approval_wait_survives_restart_and_completion_cannot_unblock_it(repository):
    run = claim(repository, enqueue(repository))
    task, _ = checkpoint(
        repository,
        run,
        status="waiting",
        wait_kind="user",
        blocker="Need approval",
        message="May I publish the report?",
    )
    complete(repository, run)
    restarted = Repository(repository.data_dir)
    restarted.initialize()
    completion = event(restarted, run)
    assert completion.status == "cancelled"
    assert restarted.wake_tasks() == []
    assert restarted.get_task("alpha", task.id).wait_kind == "user"
    resource = restarted.get_resource("alpha", run.conversation_id)
    reply = restarted.create_run(
        "alpha", resource.id, message(content="Approved"), resource.version
    )
    assert reply.trigger_message_id is not None
    assert "Finished the draft" in reply.trigger_event.content
    assert claim(restarted, reply).task_id == task.id


def test_completion_between_user_reply_and_claim_is_retained(repository):
    run = claim(repository, enqueue(repository))
    checkpoint(
        repository,
        run,
        status="waiting",
        wait_kind="user",
        blocker="Need approval",
        message="May I publish?",
    )
    complete(repository, run)
    resource = repository.get_resource("alpha", run.conversation_id)
    reply = repository.create_run(
        "alpha", resource.id, message(content="Yes"), resource.version
    )
    assert reply.trigger_event is None
    assert event(repository, run).status == "cancelled"
    assert "Finished the draft" in claim(repository, reply).trigger_event.content


def test_timer_wakeup_enqueue_rolls_back_atomically(repository, monkeypatch):
    run = claim(repository, enqueue(repository))
    now = utc_now()
    task, _ = checkpoint(
        repository,
        run,
        status="waiting",
        wait_kind="timer",
        wake_at=now + timedelta(seconds=40),
    )
    complete(repository, run)
    monkeypatch.setattr(
        "tsuchi.repository.utc_now", lambda: now + timedelta(seconds=45)
    )
    with monkeypatch.context() as interruption:

        def interrupted(*args, **kwargs):
            raise RuntimeError("Process stopped before wake state saved")

        interruption.setattr(Repository, "_task_change", staticmethod(interrupted))
        with pytest.raises(RuntimeError):
            repository.wake_tasks()
    assert repository.active_runs() == []
    restarted = Repository(repository.data_dir)
    assert len(restarted.wake_tasks()) == 1
    assert restarted.get_task("alpha", task.id).status == "active"


def test_separate_scheduler_instances_share_one_durable_wakeup(repository, monkeypatch):
    run = claim(repository, enqueue(repository))
    now = utc_now()
    task, _ = checkpoint(
        repository,
        run,
        status="waiting",
        wait_kind="timer",
        wake_at=now + timedelta(seconds=40),
    )
    complete(repository, run)
    first = Repository(repository.data_dir)
    second = Repository(repository.data_dir)
    assert first.get_task("alpha", task.id) == second.get_task("alpha", task.id)
    monkeypatch.setattr(
        "tsuchi.repository.utc_now", lambda: now + timedelta(seconds=45)
    )
    assert len(first.wake_tasks()) == 1
    assert second.wake_tasks() == []
    restarted = Repository(repository.data_dir)
    assert restarted.wake_tasks() == []
    assert len(restarted.active_runs()) == 1


def test_restart_after_wakeup_claim_resumes_same_run(repository, monkeypatch):
    run = claim(repository, enqueue(repository))
    now = utc_now()
    task, _ = checkpoint(
        repository,
        run,
        status="waiting",
        wait_kind="timer",
        wake_at=now + timedelta(seconds=40),
    )
    complete(repository, run)
    monkeypatch.setattr(
        "tsuchi.repository.utc_now", lambda: now + timedelta(seconds=45)
    )
    wake_task = repository.wake_tasks()[0]
    wake = claim(repository, repository.get_run("alpha", wake_task.current_run_id))
    expired = expire_run(repository, wake)
    restarted = Repository(repository.data_dir)
    resumed = claim(restarted, expired)
    assert resumed.id == wake.id
    assert restarted.wake_tasks() == []
    assert restarted.get_task("alpha", task.id).current_run_id == wake.id


def test_repeated_completion_is_one_event_run(repository):
    run = claim(repository, enqueue(repository))
    checkpoint(repository, run, status="waiting", wait_kind="tool")
    event_id = uuid4()
    completion = RunEvent(kind="codex", content="Complete")
    first = repository.create_event_run(
        "alpha", run.conversation_id, event_id, run.id, completion
    )
    again = repository.create_event_run(
        "alpha", run.conversation_id, event_id, run.id, completion
    )
    assert first == again
    assert (
        len(
            [
                r
                for r in repository.conversation_runs("alpha", run.conversation_id)
                if r.id == event_id
            ]
        )
        == 1
    )


@pytest.mark.parametrize("action", ["pause", "cancel"])
@pytest.mark.parametrize("completion_first", [False, True])
def test_control_and_completion_orderings_are_fenced(
    repository, action, completion_first
):
    run = claim(repository, enqueue(repository))
    task, _ = checkpoint(repository, run, status="waiting", wait_kind="tool")
    completion = event(repository, run) if completion_first else None
    repository.control_task("alpha", task.id, action, task.version)
    completion = completion or event(repository, run)
    assert repository.get_run("alpha", completion.id).status == "cancelled"
    assert repository.active_runs() == []


def test_stale_owner_cannot_write_after_explicit_resumption(repository):
    run = claim(repository, enqueue(repository))
    task, value = checkpoint(repository, run)
    paused = repository.control_task("alpha", task.id, "pause", task.version)
    resumed = repository.control_task("alpha", task.id, "resume", paused.version)
    wake = claim(repository, repository.get_run("alpha", resumed.current_run_id))
    with pytest.raises(StorageError):
        repository.checkpoint_task("alpha", run.id, value, run.owner_id)
    with pytest.raises(StorageError):
        repository.deliver_run_message(
            "alpha",
            run.id,
            run.owner_id,
            message(role="assistant", content="Old result"),
        )
    assert repository.get_task("alpha", task.id).current_run_id == wake.id


def test_complete_store_copy_preserves_goal_and_receipt(repository, tmp_path):
    import shutil

    run = claim(repository, enqueue(repository))
    task, value = checkpoint(
        repository,
        run,
        status="waiting",
        wait_kind="user",
        blocker="Need approval",
        message="May I proceed?",
    )
    copy = Repository(tmp_path / "restored")
    copy.data_dir.mkdir()
    shutil.copy2(repository.database_path, copy.database_path)
    copy.initialize()
    assert copy.get_task("alpha", task.id) == task
    assert copy.checkpoint_task("alpha", run.id, value, run.owner_id) == task
    assert len(copy.get_resource("alpha", run.conversation_id).messages) == 2


def test_resume_failed_goal_rejects_newer_unfinished_goal(repository):
    run = claim(repository, enqueue(repository))
    old, _ = checkpoint(repository, run)
    complete(repository, run)
    old = repository.get_task("alpha", old.id)
    resource = repository.get_resource("alpha", run.conversation_id)
    next_run = repository.create_run(
        "alpha", resource.id, message(content="Another report"), resource.version
    )
    next_run = claim(repository, next_run)
    current, _ = checkpoint(repository, next_run)
    with pytest.raises(StorageError, match="unfinished task"):
        repository.control_task("alpha", old.id, "resume", old.version)
    assert repository.get_task("alpha", current.id).status == "active"
    assert repository.get_task("alpha", old.id).status == "failed"


@pytest.mark.parametrize("recorded_result", [False, True])
def test_resume_includes_pending_delegation_identity_without_reasoning(
    repository, recorded_result
):
    import json
    from uuid import uuid5

    from tsuchi.models import RunMessage

    run = claim(repository, enqueue(repository))
    task, _ = checkpoint(repository, run)
    thread_id = str(uuid4())
    transcript = [
        RunMessage(role="assistant", content="Do not expose hidden reasoning"),
        RunMessage(
            role="assistant",
            content=json.dumps(
                {
                    "tool": "codex_start",
                    "arguments": {
                        "prompt": "Write the report in the existing project",
                        "cwd": "/work/reports",
                    },
                }
            ),
        ),
    ]
    if recorded_result:
        transcript.append(
            RunMessage(
                role="user",
                content="Tool result: "
                + json.dumps({"thread_id": thread_id, "running": True}),
            )
        )
    run = repository.get_run("alpha", run.id)
    run = repository.update_run(
        "alpha",
        run.id,
        run_state(run, transcript=transcript),
        run.version,
        run.owner_id,
    )
    task = repository.control_task("alpha", task.id, "pause", task.version)
    resumed = repository.control_task("alpha", task.id, "resume", task.version)
    wake = repository.get_run("alpha", resumed.current_run_id)
    assert "Write the report" in wake.trigger_event.content
    assert str(uuid5(run.id, "command:2")) in wake.trigger_event.content
    assert "Do not expose hidden reasoning" not in wake.trigger_event.content
    if recorded_result:
        assert thread_id in wake.trigger_event.content
        assert "A tool result was recorded" in wake.trigger_event.content
    else:
        assert "May have started" in wake.trigger_event.content


def test_event_queued_before_question_cannot_clear_approval_wait(repository):
    run = claim(repository, enqueue(repository))
    task, _ = checkpoint(repository, run)
    completion = event(repository, run)
    task, _ = checkpoint(
        repository,
        run,
        task,
        status="waiting",
        wait_kind="user",
        blocker="Need approval",
        message="May I publish?",
    )
    complete(repository, run)
    assert repository.get_run("alpha", completion.id).status == "cancelled"
    assert repository.get_task("alpha", task.id).wait_kind == "user"
    assert repository.active_runs() == []
    resource = repository.get_resource("alpha", run.conversation_id)
    reply = repository.create_run(
        "alpha", resource.id, message(content="Approved"), resource.version
    )
    assert "Finished the draft" in reply.trigger_event.content


def test_question_keeps_already_queued_user_reply_linked(repository):
    run = claim(repository, enqueue(repository))
    task, _ = checkpoint(repository, run)
    completion = event(repository, run)
    resource = repository.get_resource("alpha", run.conversation_id)
    reply = repository.create_run(
        "alpha", resource.id, message(content="Approved"), resource.version
    )
    task, _ = checkpoint(
        repository,
        run,
        task,
        status="waiting",
        wait_kind="user",
        blocker="Need approval",
        message="May I publish?",
    )
    complete(repository, run)
    assert repository.get_run("alpha", completion.id).status == "cancelled"
    assert repository.get_run("alpha", reply.id).task_id == task.id
    assert claim(repository, reply).task_id == task.id


def test_concurrent_schedulers_enqueue_one_wakeup(repository, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    run = claim(repository, enqueue(repository))
    now = utc_now()
    task, _ = checkpoint(
        repository,
        run,
        status="waiting",
        wait_kind="timer",
        wake_at=now + timedelta(seconds=40),
    )
    complete(repository, run)
    monkeypatch.setattr(
        "tsuchi.repository.utc_now", lambda: now + timedelta(seconds=45)
    )
    schedulers = [Repository(repository.data_dir), Repository(repository.data_dir)]
    barrier = Barrier(2)

    def wake(scheduler):
        barrier.wait(timeout=5)
        return scheduler.wake_tasks()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(wake, schedulers))
    assert sorted(len(result) for result in results) == [0, 1]
    assert len([run for run in repository.active_runs() if run.task_id == task.id]) == 1
