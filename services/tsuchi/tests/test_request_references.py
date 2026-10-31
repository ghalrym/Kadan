from uuid import uuid4

import pytest
from pydantic import ValidationError
from tsuchi.config import StorageError
from tsuchi.models import (
    AgentInput,
    Asset,
    ConversationContent,
    MarkdownContent,
    Message,
    RequestReference,
    utc_now,
)
from tsuchi.repository import Repository
from tsuchi.routes import RunCreate


@pytest.fixture
def setup(tmp_path):
    repo = Repository(tmp_path)
    repo.initialize()
    for key in ("alpha", "beta"):
        repo.create_agent(AgentInput(key=key, name=key))
    chat = repo.create_resource(
        "alpha", ConversationContent(kind="conversation", title="Chat")
    )
    return repo, chat


def reference(repo, key="alpha", kind="space", markdown="Saved text"):
    return repo.create_resource(
        key, MarkdownContent(kind=kind, title="Plan", markdown=markdown)
    )


def submit(repo, chat, refs, message=None):
    return repo.create_run(
        "alpha",
        chat.id,
        message
        or Message(id=uuid4(), created_at=utc_now(), role="user", content="Read this"),
        chat.version,
        references=[
            RequestReference(resource_id=r.id, expected_version=r.version) for r in refs
        ],
    )


def test_snapshot_is_immutable_and_retry_uses_original_after_trash(setup):
    repo, chat = setup
    source = reference(repo)
    msg = Message(id=uuid4(), created_at=utc_now(), role="user", content="Use this")
    run = submit(repo, chat, [source], msg)
    repo.trash_resource("alpha", source.id, source.version)
    assert submit(repo, chat, [source], msg) == run
    assert run.references[0].markdown == "Saved text"
    assert run.references[0].version == source.version
    assert repo.get_run("alpha", run.id).references == run.references
    with pytest.raises(StorageError):
        submit(repo, chat, [], msg)


@pytest.mark.parametrize(
    "failure", ["stale", "scope", "trash", "kind", "budget", "duplicate"]
)
def test_rejects_before_message_is_appended(setup, failure):
    repo, chat = setup
    source = reference(
        repo,
        key="beta" if failure == "scope" else "alpha",
        kind="persona" if failure == "kind" else "memory",
        markdown="x" * 12001 if failure == "budget" else "text",
    )
    if failure == "trash":
        repo.trash_resource("alpha", source.id, source.version)
    if failure == "stale":
        source = source.model_copy(update={"version": source.version + 1})
    with pytest.raises(StorageError):
        submit(repo, chat, [source, source] if failure == "duplicate" else [source])
    assert repo.get_resource("alpha", chat.id).messages == []


def test_reference_schema_bounds():
    with pytest.raises(ValidationError):
        RequestReference(resource_id=uuid4(), expected_version=0)
    with pytest.raises(ValidationError):
        RunCreate(
            message=Message(
                id=uuid4(), created_at=utc_now(), role="user", content="Hi"
            ),
            expected_version=1,
            references=[RequestReference(resource_id=uuid4(), expected_version=1)] * 4,
        )


def test_retry_after_edit_keeps_original_snapshot(setup):
    repo, chat = setup
    source = reference(repo)
    msg = Message(id=uuid4(), created_at=utc_now(), role="user", content="Use this")
    run = submit(repo, chat, [source], msg)
    updated = repo.update_resource(
        "alpha",
        source.id,
        MarkdownContent(kind="space", title="Renamed", markdown="Changed"),
        source.version,
    )
    assert submit(repo, chat, [source], msg).references == run.references
    with pytest.raises(StorageError):
        submit(repo, chat, [updated], msg)


def test_combined_reference_budget_rejects_atomically(setup):
    repo, chat = setup
    sources = [reference(repo, markdown="x" * 7000) for _ in range(2)]
    with pytest.raises(StorageError):
        submit(repo, chat, sources)
    assert not repo.get_resource("alpha", chat.id).messages


def test_image_event_inherits_request_snapshots(setup):
    from tsuchi.models import ImageJob

    repo, chat = setup
    run = submit(repo, chat, [reference(repo)])
    now = utc_now()
    job = ImageJob(
        id=uuid4(),
        agent_key="alpha",
        run_id=run.id,
        conversation_id=chat.id,
        prompt="Draw",
        request_id=uuid4(),
        event_run_id=uuid4(),
        version=1,
        created_at=now,
        updated_at=now,
    )
    with repo.connect(write=True) as db:
        repo._image_event(db, job)
    assert repo.get_run("alpha", job.event_run_id).references == run.references


def test_three_references_at_exact_combined_limit_keep_source_metadata(setup):
    repo, chat = setup
    sources = [
        reference(repo, kind=kind, markdown="x" * 4000)
        for kind in ("space", "memory", "space")
    ]
    run = submit(repo, chat, sources)
    assert [
        (r.resource_id, r.kind, r.title, r.version, r.markdown) for r in run.references
    ] == [(r.id, r.kind, r.title, r.version, r.markdown) for r in sources]
    assert repo.get_run("alpha", run.id) == run


def test_save_failure_rolls_back_message_and_run_with_references(setup, monkeypatch):
    repo, chat = setup
    source = reference(repo)
    save_run = repo._save_run

    def fail_after_save(db, run):
        save_run(db, run)
        raise RuntimeError("Failed before commit")

    monkeypatch.setattr(repo, "_save_run", fail_after_save)
    with pytest.raises(RuntimeError, match="before commit"):
        submit(repo, chat, [source])
    assert repo.get_resource("alpha", chat.id) == chat
    assert repo.conversation_runs("alpha", chat.id) == []


def test_reference_validation_and_retry_coexist_with_attachments(setup, tmp_path):
    repo, chat = setup
    source = reference(repo)
    msg = Message(
        id=uuid4(),
        created_at=utc_now(),
        role="user",
        content="Continue",
        attachment_ids=[uuid4()],
    )
    with pytest.raises(StorageError):
        repo.create_run(
            "alpha",
            chat.id,
            msg,
            chat.version,
            references=[
                RequestReference(resource_id=source.id, expected_version=source.version)
            ],
        )
    assert repo.get_resource("alpha", chat.id) == chat
    assert repo.conversation_runs("alpha", chat.id) == []
    now = utc_now()
    asset = repo.create_asset(
        Asset(
            id=uuid4(),
            agent_key="alpha",
            title="File",
            filename="file.txt",
            media_type="text/plain",
            size_bytes=3,
            sha256="abc",
            is_image=False,
            version=1,
            created_at=now,
            updated_at=now,
        )
    )
    msg = msg.model_copy(update={"attachment_ids": [asset.id]})
    run = repo.create_run(
        "alpha",
        chat.id,
        msg,
        chat.version,
        references=[
            RequestReference(resource_id=source.id, expected_version=source.version)
        ],
    )
    repo.trash_resource("alpha", source.id, source.version)
    restarted = Repository(tmp_path)
    restarted.initialize()
    assert (
        restarted.create_run(
            "alpha",
            chat.id,
            msg,
            chat.version,
            references=[
                RequestReference(resource_id=source.id, expected_version=source.version)
            ],
        )
        == run
    )
