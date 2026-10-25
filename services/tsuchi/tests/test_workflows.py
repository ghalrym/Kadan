from uuid import uuid4

import pytest
from tsuchi.config import StorageError
from tsuchi.models import (
    AgentInput,
    ConversationContent,
    MarkdownContent,
    Message,
    utc_now,
)
from tsuchi.repository import Repository


@pytest.fixture
def library(tmp_path):
    repo = Repository(tmp_path)
    repo.initialize()
    repo.create_agent(AgentInput(key="alpha", name="Alpha"))
    repo.create_agent(AgentInput(key="beta", name="Beta"))
    chat = repo.create_resource(
        "alpha", ConversationContent(kind="conversation", title="Chat")
    )
    workflow = repo.create_resource(
        "alpha",
        MarkdownContent(
            kind="space",
            title="Review",
            tags=["workflow"],
            markdown="Check the current task.",
        ),
    )
    return repo, chat, workflow


def run(library, **changes):
    repo, chat, workflow = library
    return repo.create_run(
        "alpha",
        chat.id,
        changes.pop(
            "message",
            Message(
                id=uuid4(),
                created_at=utc_now(),
                content="Review this change",
                role="user",
            ),
        ),
        changes.pop("expected_version", chat.version),
        workflow_id=changes.pop("workflow_id", workflow.id),
        workflow_version=changes.pop("workflow_version", workflow.version),
        **changes,
    )


def test_snapshot_and_current_user_task_are_distinct(library):
    repo, chat, workflow = library
    queued = run(library)
    assert queued.workflow.model_dump() == {
        "id": workflow.id,
        "title": workflow.title,
        "version": 1,
        "instructions": workflow.markdown,
    }
    saved = repo.get_resource("alpha", chat.id)
    assert saved.messages[-1].role == "user"
    assert saved.messages[-1].content == "Review this change"
    assert workflow.markdown not in saved.messages[-1].content


@pytest.mark.parametrize(
    "case",
    [
        "other_agent",
        "missing",
        "tag",
        "kind",
        "trashed",
        "version",
        "oversize",
        "blank",
    ],
)
def test_invalid_selection_does_not_append_task_or_create_run(library, case):
    repo, chat, workflow = library
    changes = {}
    if case == "other_agent":
        foreign = repo.create_resource(
            "beta",
            MarkdownContent(
                kind="space", title="Other", markdown="Other", tags=["workflow"]
            ),
        )
        changes["workflow_id"] = foreign.id
    elif case == "missing":
        changes["workflow_id"] = uuid4()
    elif case == "trashed":
        repo.trash_resource("alpha", workflow.id, workflow.version)
    elif case == "version":
        changes["workflow_version"] = 2
    else:
        content = MarkdownContent(
            kind="memory" if case == "kind" else "space",
            title="Bad",
            tags=[] if case == "tag" else ["workflow"],
            markdown="x" * 12001
            if case == "oversize"
            else " "
            if case == "blank"
            else "Method",
        )
        bad = repo.create_resource("alpha", content)
        changes["workflow_id"] = bad.id
    with pytest.raises(StorageError):
        run(library, **changes)
    assert repo.get_resource("alpha", chat.id).messages == []
    assert repo.conversation_runs("alpha", chat.id) == []


@pytest.mark.parametrize(
    "role,body", [("assistant", "Do work"), ("tool", "Do work"), ("user", " ")]
)
def test_workflow_requires_nonblank_current_user_message(library, role, body):
    with pytest.raises(StorageError):
        run(
            library,
            message=Message(id=uuid4(), created_at=utc_now(), role=role, content=body),
        )


@pytest.mark.parametrize("workflow_id,version", [(None, 1), (uuid4(), None)])
def test_workflow_selection_requires_id_and_version(library, workflow_id, version):
    with pytest.raises(StorageError):
        run(library, workflow_id=workflow_id, workflow_version=version)


def test_retry_keeps_snapshot_after_revision_and_trash(library):
    repo, chat, workflow = library
    message = Message(
        id=uuid4(), created_at=utc_now(), role="user", content="Review this change"
    )
    first = run(library, message=message)
    revised = repo.update_resource(
        "alpha",
        workflow.id,
        MarkdownContent(
            kind="space",
            title="New title",
            tags=["workflow"],
            markdown="Different method",
        ),
        1,
    )
    repo.trash_resource("alpha", workflow.id, revised.version)
    retry = run(library, message=message)
    assert retry == first
    assert retry.workflow.instructions == workflow.markdown
    assert len(repo.get_resource("alpha", chat.id).messages) == 1
    with pytest.raises(StorageError):
        run(library, message=message, workflow_version=revised.version)
    with pytest.raises(StorageError):
        run(library, message=message, workflow_id=None, workflow_version=None)


def test_new_run_uses_revised_snapshot_and_ordinary_send_uses_none(library):
    repo, chat, workflow = library
    first = run(library)
    revised = repo.update_resource(
        "alpha",
        workflow.id,
        MarkdownContent(
            kind="space",
            title="New title",
            tags=["workflow"],
            markdown="Different method",
        ),
        1,
    )
    current_chat = repo.get_resource("alpha", chat.id)
    second = run((repo, current_chat, revised))
    assert first.workflow.instructions != second.workflow.instructions
    ordinary = run(
        (repo, repo.get_resource("alpha", chat.id), revised),
        workflow_id=None,
        workflow_version=None,
    )
    assert ordinary.workflow is None


def test_workflow_and_references_share_one_stable_request_snapshot(library):
    from tsuchi.models import RequestReference

    repo, chat, workflow = library
    reference = repo.create_resource(
        "alpha",
        MarkdownContent(kind="memory", title="Evidence", markdown="Original fact"),
    )
    message = Message(
        id=uuid4(), created_at=utc_now(), role="user", content="Review this task"
    )
    selection = [
        RequestReference(resource_id=reference.id, expected_version=reference.version)
    ]
    first = repo.create_run(
        "alpha",
        chat.id,
        message,
        chat.version,
        references=selection,
        workflow_id=workflow.id,
        workflow_version=workflow.version,
    )
    for source in (reference, workflow):
        revised = repo.update_resource(
            "alpha",
            source.id,
            MarkdownContent(
                kind=source.kind,
                title=source.title,
                tags=source.tags,
                markdown="Later edit",
            ),
            source.version,
        )
        repo.trash_resource("alpha", source.id, revised.version)
    recovered = repo.create_run(
        "alpha",
        chat.id,
        message,
        chat.version,
        references=selection,
        workflow_id=workflow.id,
        workflow_version=workflow.version,
    )
    assert recovered == first
    assert recovered.references[0].markdown == "Original fact"
    assert recovered.workflow.instructions == workflow.markdown
    saved = repo.get_resource("alpha", chat.id)
    assert len(saved.messages) == 1
    assert saved.messages[0].role == "user"
    assert saved.messages[0].content == message.content
    with pytest.raises(StorageError):
        repo.create_run(
            "alpha",
            chat.id,
            message,
            chat.version,
            references=[],
            workflow_id=workflow.id,
            workflow_version=workflow.version,
        )


def test_invalid_workflow_with_valid_references_writes_no_partial_request(library):
    from tsuchi.models import RequestReference

    repo, chat, workflow = library
    selection = [
        RequestReference(resource_id=workflow.id, expected_version=workflow.version)
    ]
    with pytest.raises(StorageError):
        run(library, references=selection, workflow_version=workflow.version + 1)
    assert repo.get_resource("alpha", chat.id).messages == []
    assert repo.conversation_runs("alpha", chat.id) == []
