from datetime import timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError
from tsuchi.config import StorageError
from tsuchi.models import (
    AgentInput,
    AgentRunState,
    Asset,
    ConversationContent,
    ImageJobState,
    MarkdownContent,
    Message,
    RunMessage,
    utc_now,
)
from tsuchi.repository import Repository


@pytest.fixture
def repository(tmp_path):
    repository = Repository(tmp_path)
    repository.initialize()
    repository.create_agent(AgentInput(key="alpha", name="Alpha"))
    repository.create_agent(AgentInput(key="beta", name="Beta"))
    return repository


def conversation(repository, key="alpha"):
    return repository.create_resource(
        key, ConversationContent(kind="conversation", title="A conversation")
    )


def message(role="user", content="Draw a fox", **kwargs):
    return Message(
        id=uuid4(), role=role, content=content, created_at=utc_now(), **kwargs
    )


def enqueue(repository, resource=None):
    resource = resource or conversation(repository)
    return repository.create_run("alpha", resource.id, message(), resource.version)


def run_state(run, **changes):
    return AgentRunState.model_validate(
        {**run.model_dump(include=set(AgentRunState.model_fields)), **changes}
    )


def image_state(job, **changes):
    return ImageJobState.model_validate(
        {**job.model_dump(include=set(ImageJobState.model_fields)), **changes}
    )


def claim(repository, run, owner=None):
    runners = repository.list_runners()
    current = runners[0] if runners else None
    owner = owner or (
        current.worker_id
        if current and (run.status != "running" or run.lease_expires_at > utc_now())
        else uuid4()
    )
    repository.register_runner(
        "test", owner, expected_worker_id=current.worker_id if current else None
    )
    repository.assign_runner("test", owner)
    run = repository.get_run("alpha", run.id)
    return repository.update_run(
        "alpha",
        run.id,
        run_state(
            run,
            status="running",
            owner_id=owner,
            lease_expires_at=utc_now() + timedelta(seconds=60),
        ),
        run.version,
        owner,
    )


def image_job(repository, run=None):
    run = run or claim(repository, enqueue(repository))
    return repository.create_image_job(
        "alpha", run.id, uuid4(), "A fox", uuid4(), run.owner_id
    )


def claim_image(repository, job, owner=None):
    owner = owner or uuid4()
    return repository.update_image_job(
        "alpha",
        job.id,
        image_state(
            job,
            status="cancelling" if job.status == "cancelling" else "running",
            owner_id=owner,
            lease_expires_at=utc_now() + timedelta(seconds=60),
        ),
        job.version,
        owner,
    )


def image_asset(repository, key="alpha", is_image=True):
    now = utc_now()
    return repository.create_asset(
        Asset(
            id=uuid4(),
            agent_key=key,
            title="Fox",
            filename="fox.png",
            media_type="image/png",
            size_bytes=12,
            sha256="digest",
            is_image=is_image,
            version=1,
            created_at=now,
            updated_at=now,
        )
    )


def complete_image(repository, job):
    return repository.update_image_job(
        "alpha",
        job.id,
        image_state(
            job,
            status="completed",
            owner_id=None,
            lease_expires_at=None,
            asset_id=image_asset(repository).id,
        ),
        job.version,
        job.owner_id,
    )


def expire_run(repository, run):
    with repository.connect(write=True) as db:
        repository._save_run(
            db,
            run.model_copy(
                update={
                    "lease_expires_at": utc_now() - timedelta(seconds=1),
                }
            ),
        )
    return repository.get_run("alpha", run.id)


def expire_image(repository, job):
    with repository.connect(write=True) as db:
        repository._save_image_job(
            db,
            job.model_copy(
                update={
                    "lease_expires_at": utc_now() - timedelta(seconds=1),
                }
            ),
        )
    return repository.get_image_job("alpha", job.id)


def test_enqueue_is_atomic_and_retry_ignores_stale_version(repository):
    resource = conversation(repository)
    user = message()
    run = repository.create_run("alpha", resource.id, user, resource.version)
    retried = repository.create_run(
        "alpha",
        resource.id,
        user.model_copy(update={"created_at": utc_now()}),
        resource.version,
    )
    assert retried == run
    assert repository.get_resource("alpha", resource.id).messages == [user]
    assert repository.active_runs() == [run]


def test_enqueue_conflict_does_not_append_or_create_run(repository):
    resource = conversation(repository)
    repository.append_message("alpha", resource.id, message(), resource.version)
    with pytest.raises(StorageError):
        repository.create_run("alpha", resource.id, message(), resource.version)
    assert len(repository.get_resource("alpha", resource.id).messages) == 1
    assert repository.active_runs() == []


@pytest.mark.parametrize("status", ["queued", "running", "cancelled"])
def test_trashed_conversation_runs_leave_dispatch_until_restored(repository, status):
    older = enqueue(repository)
    if status == "running":
        older = claim(repository, older)
        older = repository.update_run(
            "alpha",
            older.id,
            run_state(
                older,
                transcript=[RunMessage(role="assistant", content="Saved command")],
            ),
            older.version,
            older.owner_id,
        )
    elif status == "cancelled":
        older = repository.cancel_run("alpha", older.id)
    resource = repository.get_resource("alpha", older.conversation_id)
    trashed = repository.trash_resource("alpha", resource.id, resource.version)
    newer = enqueue(repository)

    assert repository.active_runs() == [newer]
    assert repository.get_run("alpha", older.id) == older
    repository.restore_resource("alpha", resource.id, trashed.version)
    assert repository.active_runs() == (
        [newer] if status == "cancelled" else [older, newer]
    )
    assert repository.get_run("alpha", older.id) == older


@pytest.mark.parametrize("parent", ["other_agent", "memory"])
def test_dispatch_requires_an_active_conversation_in_the_run_namespace(
    repository, parent
):
    run = enqueue(repository)
    invalid_parent = (
        conversation(repository, "beta")
        if parent == "other_agent"
        else repository.create_resource(
            "alpha", MarkdownContent(kind="memory", title="Not a conversation")
        )
    )
    invalid = run.model_copy(update={"conversation_id": invalid_parent.id})
    # Represent an invalid stored reference; normal enqueue rejects these parents.
    with repository.connect(write=True) as db:
        db.execute(
            "UPDATE agent_runs SET conversation_id = ?, data = ? WHERE id = ?",
            (str(invalid_parent.id), invalid.model_dump_json(), str(run.id)),
        )
    assert repository.active_runs() == []
    assert repository.get_run("alpha", run.id) == invalid


def test_run_id_cannot_be_reused_with_different_input(repository):
    resource = conversation(repository)
    user = message()
    repository.create_run("alpha", resource.id, user, resource.version)
    with pytest.raises(StorageError, match="already in use"):
        repository.create_run(
            "alpha", resource.id, user.model_copy(update={"content": "Different"}), 2
        )


def test_only_one_live_run_can_own_a_conversation(repository):
    first = claim(repository, enqueue(repository))
    second = enqueue(
        repository, repository.get_resource("alpha", first.conversation_id)
    )
    with pytest.raises(StorageError, match="Another run owns"):
        claim(repository, second)
    assert repository.get_run("alpha", second.id).status == "queued"
    # Different conversations can make progress independently.
    assert claim(repository, enqueue(repository)).status == "running"


def test_expired_run_recovery_fences_previous_owner(repository):
    first = claim(repository, enqueue(repository))
    persisted = repository.update_run(
        "alpha",
        first.id,
        run_state(
            first, transcript=[RunMessage(role="assistant", content="Saved action")]
        ),
        first.version,
        first.owner_id,
    )
    stale = expire_run(repository, persisted)
    recovered = claim(repository, stale)
    assert recovered.transcript == persisted.transcript
    with pytest.raises(StorageError, match="not owned"):
        repository.deliver_run_message(
            "alpha", recovered.id, first.owner_id, message("assistant")
        )


def test_expired_older_run_recovers_before_newer_queued_run(repository):
    older = claim(repository, enqueue(repository))
    newer = enqueue(repository, repository.get_resource("alpha", older.conversation_id))
    # The old worker's last heartbeat happened after the newer message arrived.
    older = repository.update_run(
        "alpha",
        older.id,
        run_state(older, lease_expires_at=utc_now() + timedelta(seconds=60)),
        older.version,
        older.owner_id,
    )
    older = expire_run(repository, older)
    assert older.updated_at > newer.updated_at
    assert [run.id for run in repository.active_runs()] == [older.id, newer.id]
    with pytest.raises(StorageError, match="earlier run"):
        claim(repository, newer)
    recovered = claim(repository, older)
    repository.update_run(
        "alpha",
        recovered.id,
        run_state(recovered, status="completed", owner_id=None, lease_expires_at=None),
        recovered.version,
        recovered.owner_id,
    )
    assert claim(repository, newer).status == "running"


def test_newer_run_cannot_bypass_older_queued_run(repository):
    older = enqueue(repository)
    newer = enqueue(repository, repository.get_resource("alpha", older.conversation_id))
    with pytest.raises(StorageError, match="earlier run"):
        claim(repository, newer)
    assert claim(repository, older).status == "running"


def test_run_order_uses_uuid_to_break_creation_timestamp_ties(repository):
    first = enqueue(repository)
    second = enqueue(
        repository, repository.get_resource("alpha", first.conversation_id)
    )
    second = second.model_copy(update={"created_at": first.created_at})
    with repository.connect(write=True) as db:
        repository._save_run(db, second)
    earlier, later = sorted([first, second], key=lambda run: run.id)
    assert [run.id for run in repository.active_runs()] == [earlier.id, later.id]
    with pytest.raises(StorageError, match="earlier run"):
        claim(repository, later)
    assert claim(repository, earlier).status == "running"


def test_recovery_cannot_change_persisted_transcript(repository):
    run = expire_run(repository, claim(repository, enqueue(repository)))
    owner = run.owner_id
    with pytest.raises(StorageError, match="cannot change"):
        repository.update_run(
            "alpha",
            run.id,
            run_state(
                run,
                owner_id=owner,
                lease_expires_at=utc_now() + timedelta(seconds=60),
                transcript=[RunMessage(role="assistant", content="Invented")],
            ),
            run.version,
            owner,
        )


def test_transcript_updates_cannot_rewrite_previous_actions(repository):
    run = claim(repository, enqueue(repository))
    run = repository.update_run(
        "alpha",
        run.id,
        run_state(run, transcript=[RunMessage(role="assistant", content="Action")]),
        run.version,
        run.owner_id,
    )
    with pytest.raises(StorageError, match="immutable"):
        repository.update_run(
            "alpha", run.id, run_state(run, transcript=[]), run.version, run.owner_id
        )


def test_message_replay_is_idempotent_even_after_run_finishes(repository):
    run = claim(repository, enqueue(repository))
    output = message("assistant", "Here is an update")
    delivered = repository.deliver_run_message("alpha", run.id, run.owner_id, output)
    current = repository.get_run("alpha", run.id)
    repository.update_run(
        "alpha",
        run.id,
        run_state(current, status="completed", owner_id=None, lease_expires_at=None),
        current.version,
        run.owner_id,
    )
    replay = repository.deliver_run_message(
        "alpha",
        run.id,
        run.owner_id,
        output.model_copy(update={"created_at": utc_now()}),
    )
    assert replay == delivered
    assert replay.messages[-1] == output
    assert len(replay.messages) == 2


def test_finishing_run_adds_no_visible_message(repository):
    run = claim(repository, enqueue(repository))
    repository.update_run(
        "alpha",
        run.id,
        run_state(run, status="completed", owner_id=None, lease_expires_at=None),
        run.version,
        run.owner_id,
    )
    assert len(repository.get_resource("alpha", run.conversation_id).messages) == 1


def test_cancel_run_fences_delivery_and_does_not_delete_conversation(repository):
    run = claim(repository, enqueue(repository))
    cancelled = repository.cancel_run("alpha", run.id)
    assert repository.cancel_run("alpha", run.id) == cancelled
    with pytest.raises(StorageError, match="not owned"):
        repository.deliver_run_message(
            "alpha", run.id, run.owner_id, message("assistant")
        )
    assert len(repository.get_resource("alpha", run.conversation_id).messages) == 1


def test_image_submission_is_idempotent_and_has_no_completion_event(repository):
    run = claim(repository, enqueue(repository))
    job = image_job(repository, run)
    assert (
        repository.create_image_job(
            "alpha", run.id, job.id, job.prompt, job.request_id, run.owner_id
        )
        == job
    )
    assert repository.active_image_jobs() == [job]
    assert repository.conversation_runs("alpha", run.conversation_id) == [run]


def test_terminal_image_and_event_are_atomic_and_retry_safe(repository):
    job = claim_image(repository, image_job(repository))
    completed = complete_image(repository, job)
    repeated = repository.update_image_job(
        "alpha", job.id, image_state(completed), job.version, job.owner_id
    )
    assert repeated == completed
    event = repository.get_run("alpha", job.event_run_id)
    assert event.trigger_job_id == job.id
    assert event.conversation_id == job.conversation_id
    assert len(repository.conversation_runs("alpha", job.conversation_id)) == 2


def test_image_completion_requires_owned_namespace_image(repository):
    job = claim_image(repository, image_job(repository))
    for asset in (
        image_asset(repository, "beta"),
        image_asset(repository, is_image=False),
    ):
        with pytest.raises(StorageError):
            repository.update_image_job(
                "alpha",
                job.id,
                image_state(
                    job,
                    status="completed",
                    owner_id=None,
                    lease_expires_at=None,
                    asset_id=asset.id,
                ),
                job.version,
                job.owner_id,
            )
    assert repository.get_image_job("alpha", job.id).status == "running"
    with pytest.raises(StorageError):
        repository.get_run("alpha", job.event_run_id)


def test_duplicate_image_delivery_from_later_run_does_not_append(repository):
    origin = claim(repository, enqueue(repository))
    job = complete_image(
        repository, claim_image(repository, image_job(repository, origin))
    )
    output = message("assistant", "Your fox")
    delivered = repository.deliver_run_message(
        "alpha", origin.id, origin.owner_id, output, job.id
    )
    current = repository.get_run("alpha", origin.id)
    repository.update_run(
        "alpha",
        origin.id,
        run_state(current, status="completed", owner_id=None, lease_expires_at=None),
        current.version,
        origin.owner_id,
    )
    event = claim(repository, repository.get_run("alpha", job.event_run_id))
    repeated = repository.deliver_run_message(
        "alpha",
        event.id,
        event.owner_id,
        message("assistant", "Another delivery", attachment_ids=[job.asset_id]),
        job.id,
    )
    assert repeated == delivered
    assert delivered.messages[-1].attachment_ids == [job.asset_id]
    assert repository.get_image_job("alpha", job.id).delivered_message_id == output.id


def test_enqueue_rolls_back_message_when_run_persistence_fails(repository, monkeypatch):
    resource = conversation(repository)

    def fail(*args):
        raise RuntimeError("Storage failure")

    monkeypatch.setattr(repository, "_save_run", fail)
    with pytest.raises(RuntimeError):
        enqueue(repository, resource)
    assert repository.get_resource("alpha", resource.id).messages == []
    assert repository.active_runs() == []


def test_terminal_image_rolls_back_when_event_persistence_fails(
    repository, monkeypatch
):
    job = claim_image(repository, image_job(repository))

    def fail(*args):
        raise RuntimeError("Storage failure")

    monkeypatch.setattr(repository, "_save_run", fail)
    with pytest.raises(RuntimeError):
        complete_image(repository, job)
    assert repository.get_image_job("alpha", job.id) == job


def test_message_rolls_back_when_delivery_receipt_fails(repository, monkeypatch):
    run = claim(repository, enqueue(repository))

    def fail(*args):
        raise RuntimeError("Storage failure")

    monkeypatch.setattr(repository, "_save_run", fail)
    with pytest.raises(RuntimeError):
        repository.deliver_run_message(
            "alpha", run.id, run.owner_id, message("assistant")
        )
    assert len(repository.get_resource("alpha", run.conversation_id).messages) == 1


def test_image_delivery_retry_keeps_one_attachment_and_original_timestamp(repository):
    run = claim(repository, enqueue(repository))
    job = complete_image(
        repository, claim_image(repository, image_job(repository, run))
    )
    output = message("assistant", "Your fox")
    delivered = repository.deliver_run_message(
        "alpha", run.id, run.owner_id, output, job.id
    )
    replayed = repository.deliver_run_message(
        "alpha",
        run.id,
        run.owner_id,
        output.model_copy(update={"created_at": utc_now()}),
        job.id,
    )
    assert replayed == delivered
    assert replayed.messages[-1].created_at == output.created_at
    assert replayed.messages[-1].attachment_ids == [job.asset_id]


def test_image_cannot_be_delivered_to_another_conversation(repository):
    job = complete_image(repository, claim_image(repository, image_job(repository)))
    other = claim(repository, enqueue(repository))
    with pytest.raises(StorageError, match="this conversation"):
        repository.deliver_run_message(
            "alpha",
            other.id,
            other.owner_id,
            message("assistant", attachment_ids=[job.asset_id]),
            job.id,
        )


def test_pending_image_does_not_block_new_user_message(repository):
    origin = claim(repository, enqueue(repository))
    job = claim_image(repository, image_job(repository, origin))
    repository.update_run(
        "alpha",
        origin.id,
        run_state(origin, status="completed", owner_id=None, lease_expires_at=None),
        origin.version,
        origin.owner_id,
    )
    next_run = enqueue(
        repository, repository.get_resource("alpha", origin.conversation_id)
    )
    assert claim(repository, next_run).conversation_id == job.conversation_id
    assert repository.get_image_job("alpha", job.id).status == "running"


def test_running_image_cancel_requires_cleanup_before_terminal_event(repository):
    job = claim_image(repository, image_job(repository))
    cancelling = repository.cancel_image_job("alpha", job.id)
    assert cancelling.status == "cancelling"
    with pytest.raises(StorageError):
        repository.get_run("alpha", job.event_run_id)
    with pytest.raises(StorageError, match="Invalid image job"):
        complete_image(repository, cancelling)
    cancelled = repository.update_image_job(
        "alpha",
        job.id,
        image_state(
            cancelling, status="cancelled", owner_id=None, lease_expires_at=None
        ),
        cancelling.version,
        job.owner_id,
    )
    assert cancelled.status == "cancelled"
    assert repository.get_run("alpha", job.event_run_id).trigger_job_id == job.id


def test_queued_image_cancel_creates_exactly_one_event(repository):
    job = image_job(repository)
    cancelled = repository.cancel_image_job("alpha", job.id)
    assert repository.cancel_image_job("alpha", job.id) == cancelled
    assert len(repository.conversation_runs("alpha", job.conversation_id)) == 2


def test_expired_image_requires_reclaim_and_old_owner_is_fenced(repository):
    job = expire_image(repository, claim_image(repository, image_job(repository)))
    with pytest.raises(StorageError, match="Claim"):
        complete_image(repository, job)
    recovered = claim_image(repository, job)
    assert recovered.request_id == job.request_id
    with pytest.raises(StorageError, match="not owned"):
        repository.update_image_job(
            "alpha",
            job.id,
            image_state(
                recovered,
                status="failed",
                owner_id=None,
                lease_expires_at=None,
                error="Interrupted",
            ),
            recovered.version,
            job.owner_id,
        )
    failed = repository.update_image_job(
        "alpha",
        job.id,
        image_state(
            recovered,
            status="failed",
            owner_id=None,
            lease_expires_at=None,
            error="Interrupted",
        ),
        recovered.version,
        recovered.owner_id,
    )
    assert failed.status == "failed"
    assert repository.get_run("alpha", job.event_run_id).status == "queued"


def test_lease_models_reject_invalid_state():
    with pytest.raises(ValidationError):
        AgentRunState(status="running")
    with pytest.raises(ValidationError):
        AgentRunState(owner_id=uuid4())
    with pytest.raises(ValidationError):
        ImageJobState(status="completed")
    with pytest.raises(ValidationError):
        ImageJobState(status="failed", asset_id=uuid4())


def test_jobs_and_runs_are_exported_and_survive_reinitialization(repository):
    job = image_job(repository)
    repository.initialize()
    with repository.connect() as db:
        exported = repository.export_manifest(db, "alpha")
    assert exported.image_jobs == [job]
    assert exported.agent_runs[0].id == job.run_id
    with pytest.raises(StorageError):
        repository.get_image_job("beta", job.id)
    with pytest.raises(StorageError):
        repository.get_run("beta", job.run_id)


def test_image_event_waits_for_earlier_user_turn_then_claims_idle_conversation(
    repository,
):
    origin = claim(repository, enqueue(repository))
    job = claim_image(repository, image_job(repository, origin))
    repository.update_run(
        "alpha",
        origin.id,
        run_state(origin, status="completed", owner_id=None, lease_expires_at=None),
        origin.version,
        origin.owner_id,
    )
    user = enqueue(repository, repository.get_resource("alpha", origin.conversation_id))
    completed = complete_image(repository, job)
    event = repository.get_run("alpha", completed.event_run_id)
    with pytest.raises(StorageError, match="earlier run"):
        claim(repository, event)
    user = claim(repository, user)
    with pytest.raises(StorageError, match="Another run owns"):
        claim(repository, event)
    repository.update_run(
        "alpha",
        user.id,
        run_state(user, status="completed", owner_id=None, lease_expires_at=None),
        user.version,
        user.owner_id,
    )
    assert claim(repository, event).status == "running"
    assert len(repository.conversation_runs("alpha", origin.conversation_id)) == 3
