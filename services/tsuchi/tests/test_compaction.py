from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from tsuchi.config import Settings, StorageError
from tsuchi.main import create_app
from tsuchi.models import (
    AgentInput,
    ConversationCompactionUpdate,
    ConversationContent,
    MarkdownContent,
    Message,
    ResourceRecord,
    resource_input_adapter,
    utc_now,
)
from tsuchi.repository import Repository


@pytest.fixture
def repository(tmp_path):
    repo = Repository(tmp_path)
    repo.initialize()
    repo.create_agent(AgentInput(key="alpha", name="Alpha"))
    return repo


def make_conversation_with_messages(repository: Repository, count: int = 4):
    base = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    messages = [
        Message(
            id=uuid4(),
            role="user" if index % 2 == 0 else "assistant",
            content=f"Message {index}",
            created_at=base + timedelta(minutes=index),
        )
        for index in range(count)
    ]
    return repository.create_resource(
        "alpha",
        ConversationContent(
            kind="conversation",
            title="New chat",
            messages=messages,
        ),
    )


def test_compaction_persists_summary_without_mutating_messages(repository):
    conversation = make_conversation_with_messages(repository, count=4)
    anchor = conversation.messages[1].id
    compacted_at = datetime(2026, 11, 26, 13, 0, tzinfo=UTC)

    updated = repository.update_conversation_compaction(
        "alpha",
        conversation.id,
        ConversationCompactionUpdate(
            expected_version=conversation.version,
            compacted_summary="Discussed initial setup and goals.",
            compacted_through_message_id=anchor,
            compacted_at=compacted_at,
        ),
    )

    assert updated.version == conversation.version + 1
    assert updated.compacted_summary == "Discussed initial setup and goals."
    assert updated.compacted_through_message_id == anchor
    assert updated.compacted_at == compacted_at
    assert updated.messages == conversation.messages

    loaded = repository.get_resource("alpha", conversation.id)
    assert loaded.compacted_summary == "Discussed initial setup and goals."
    assert loaded.compacted_through_message_id == anchor
    assert loaded.compacted_at == compacted_at
    assert [m.content for m in loaded.messages] == [
        "Message 0",
        "Message 1",
        "Message 2",
        "Message 3",
    ]


def test_compaction_rejects_stale_version_and_backwards_anchor(repository):
    conversation = make_conversation_with_messages(repository, count=4)
    updated = repository.update_conversation_compaction(
        "alpha",
        conversation.id,
        ConversationCompactionUpdate(
            expected_version=conversation.version,
            compacted_summary="Summary through message 2",
            compacted_through_message_id=conversation.messages[2].id,
        ),
    )

    with pytest.raises(StorageError) as stale:
        repository.update_conversation_compaction(
            "alpha",
            conversation.id,
            ConversationCompactionUpdate(
                expected_version=conversation.version,
                compacted_summary="Stale write",
                compacted_through_message_id=conversation.messages[3].id,
            ),
        )
    assert stale.value.status_code == 409

    with pytest.raises(StorageError, match="backwards") as backwards:
        repository.update_conversation_compaction(
            "alpha",
            conversation.id,
            ConversationCompactionUpdate(
                expected_version=updated.version,
                compacted_summary="Moved backwards",
                compacted_through_message_id=conversation.messages[0].id,
            ),
        )
    assert backwards.value.status_code == 409

    with pytest.raises(StorageError, match="not found") as missing:
        repository.update_conversation_compaction(
            "alpha",
            conversation.id,
            ConversationCompactionUpdate(
                expected_version=updated.version,
                compacted_summary="Unknown anchor",
                compacted_through_message_id=uuid4(),
            ),
        )
    assert missing.value.status_code == 409


def test_compaction_require_idle_blocks_while_run_is_active(repository):
    conversation = make_conversation_with_messages(repository, count=2)
    run = repository.create_run(
        "alpha",
        conversation.id,
        Message(id=uuid4(), role="user", content="Still active", created_at=utc_now()),
        expected_version=conversation.version,
    )
    current = repository.get_resource("alpha", conversation.id)

    with pytest.raises(StorageError, match="active work") as blocked:
        repository.update_conversation_compaction(
            "alpha",
            conversation.id,
            ConversationCompactionUpdate(
                expected_version=current.version,
                compacted_summary="Idle summary",
                compacted_through_message_id=current.messages[-1].id,
                require_idle=True,
            ),
        )
    assert blocked.value.status_code == 409

    # Non-idle compaction (in-run budget compaction) is still allowed for earlier messages.
    in_run = repository.update_conversation_compaction(
        "alpha",
        conversation.id,
        ConversationCompactionUpdate(
            expected_version=current.version,
            compacted_summary="In-run summary",
            compacted_through_message_id=current.messages[1].id,
            require_idle=False,
        ),
    )
    assert in_run.compacted_through_message_id == current.messages[1].id

    # Once the run is terminal, require_idle succeeds.
    repository.cancel_run("alpha", run.id)
    after_idle = repository.update_conversation_compaction(
        "alpha",
        conversation.id,
        ConversationCompactionUpdate(
            expected_version=in_run.version,
            compacted_summary="Now idle summary",
            compacted_through_message_id=in_run.messages[-1].id,
            require_idle=True,
        ),
    )
    assert after_idle.compacted_summary == "Now idle summary"
    assert after_idle.compacted_through_message_id == in_run.messages[-1].id


def test_compaction_preserved_across_appends_and_metadata_updates(tmp_path):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        client.post("/api/agents", json={"key": "alpha", "name": "Alpha"})
        msg1_id = str(uuid4())
        msg2_id = str(uuid4())
        created = client.post(
            "/api/agents/alpha/resources",
            json={
                "kind": "conversation",
                "title": "New chat",
                "messages": [
                    {
                        "id": msg1_id,
                        "role": "user",
                        "content": "First message",
                        "created_at": "2026-11-26T10:00:00Z",
                    },
                    {
                        "id": msg2_id,
                        "role": "assistant",
                        "content": "First reply",
                        "created_at": "2026-11-26T10:01:00Z",
                    },
                ],
            },
        ).json()

        compacted = client.put(
            f"/api/agents/alpha/resources/{created['id']}/compaction",
            json={
                "expected_version": created["version"],
                "compacted_summary": "Summary of turn 1",
                "compacted_through_message_id": msg2_id,
            },
        )
        assert compacted.status_code == 200
        body = compacted.json()
        assert body["compacted_summary"] == "Summary of turn 1"
        assert body["compacted_through_message_id"] == msg2_id
        assert body["compacted_at"] is not None

        # Appending a new message keeps the saved compaction state.
        appended = client.post(
            f"/api/agents/alpha/resources/{created['id']}/messages",
            json={
                "expected_version": body["version"],
                "message": {
                    "id": str(uuid4()),
                    "role": "user",
                    "content": "Second message",
                    "created_at": "2026-11-26T11:00:00Z",
                },
            },
        )
        assert appended.status_code == 200
        after_append = appended.json()
        assert after_append["compacted_summary"] == "Summary of turn 1"
        assert after_append["compacted_through_message_id"] == msg2_id
        assert len(after_append["messages"]) == 3

        # Updating title via PUT /resources/{id} without compaction fields preserves them.
        renamed = client.put(
            f"/api/agents/alpha/resources/{created['id']}",
            json={
                "kind": "conversation",
                "title": "Renamed",
                "messages": after_append["messages"],
                "expected_version": after_append["version"],
            },
        )
        assert renamed.status_code == 200
        after_rename = renamed.json()
        assert after_rename["title"] == "Renamed"
        assert after_rename["compacted_summary"] == "Summary of turn 1"
        assert after_rename["compacted_through_message_id"] == msg2_id


def test_compaction_rejects_non_conversation_and_invalid_models(repository):
    space = repository.create_resource(
        "alpha", MarkdownContent(kind="space", title="Notes", markdown="hello")
    )
    with pytest.raises(StorageError, match="Only conversations") as error:
        repository.update_conversation_compaction(
            "alpha",
            space.id,
            ConversationCompactionUpdate(
                expected_version=space.version,
                compacted_summary="Invalid",
                compacted_through_message_id=uuid4(),
            ),
        )
    assert error.value.status_code == 409

    with pytest.raises(ValidationError):
        ConversationContent(
            kind="conversation",
            title="Bad",
            compacted_summary="Summary without anchor",
        )

    with pytest.raises(ValidationError):
        ConversationCompactionUpdate(
            expected_version=1,
            compacted_summary="   ",
            compacted_through_message_id=uuid4(),
        )

    with pytest.raises(ValidationError):
        ConversationCompactionUpdate(
            expected_version=1,
            compacted_summary="Valid",
            compacted_through_message_id=uuid4(),
            compacted_at=datetime(2026, 11, 26, 12, 0),
        )
