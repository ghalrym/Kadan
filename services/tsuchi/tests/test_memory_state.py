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
    ConversationMemoryStateUpdate,
    MarkdownContent,
    Message,
    utc_now,
)
from tsuchi.repository import Repository


@pytest.fixture
def repository(tmp_path):
    repo = Repository(tmp_path)
    repo.initialize()
    repo.create_agent(AgentInput(key="alpha", name="Alpha"))
    return repo


def make_conversation(repository: Repository, count: int = 4):
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
        ConversationContent(kind="conversation", title="New chat", messages=messages),
    )


def test_memory_state_advances_cursor_and_sets_snapshot_in_place(repository):
    conversation = make_conversation(repository, count=4)
    refreshed_at = datetime(2026, 11, 26, 13, 0, tzinfo=UTC)

    updated = repository.update_conversation_memory_state(
        "alpha",
        conversation.id,
        ConversationMemoryStateUpdate(
            memory_extracted_through_message_id=conversation.messages[1].id,
            core_memories_snapshot="Core memories:\n- **User**: Andrew",
            core_memories_refreshed_at=refreshed_at,
            only_if_snapshot_unset=True,
        ),
    )

    # In-place metadata update preserves version so user sends never hit 409.
    assert updated.version == conversation.version
    assert updated.memory_extracted_through_message_id == conversation.messages[1].id
    assert updated.core_memories_snapshot == "Core memories:\n- **User**: Andrew"
    assert updated.core_memories_refreshed_at == refreshed_at

    # only_if_snapshot_unset=True keeps existing snapshot while advancing cursor.
    second = repository.update_conversation_memory_state(
        "alpha",
        conversation.id,
        ConversationMemoryStateUpdate(
            memory_extracted_through_message_id=conversation.messages[3].id,
            core_memories_snapshot="Should not overwrite",
            only_if_snapshot_unset=True,
        ),
    )
    assert second.memory_extracted_through_message_id == conversation.messages[3].id
    assert second.core_memories_snapshot == "Core memories:\n- **User**: Andrew"
    assert second.core_memories_refreshed_at == refreshed_at


def test_memory_state_rejects_backwards_or_missing_cursor_and_active_runs(repository):
    conversation = make_conversation(repository, count=4)
    repository.update_conversation_memory_state(
        "alpha",
        conversation.id,
        ConversationMemoryStateUpdate(
            memory_extracted_through_message_id=conversation.messages[2].id,
        ),
    )

    with pytest.raises(StorageError, match="backwards") as backwards:
        repository.update_conversation_memory_state(
            "alpha",
            conversation.id,
            ConversationMemoryStateUpdate(
                memory_extracted_through_message_id=conversation.messages[0].id,
            ),
        )
    assert backwards.value.status_code == 409

    with pytest.raises(StorageError, match="not found") as missing:
        repository.update_conversation_memory_state(
            "alpha",
            conversation.id,
            ConversationMemoryStateUpdate(
                memory_extracted_through_message_id=uuid4(),
            ),
        )
    assert missing.value.status_code == 409

    repository.create_run(
        "alpha",
        conversation.id,
        Message(id=uuid4(), role="user", content="Active", created_at=utc_now()),
        expected_version=conversation.version,
    )
    with pytest.raises(StorageError, match="active work") as active:
        repository.update_conversation_memory_state(
            "alpha",
            conversation.id,
            ConversationMemoryStateUpdate(
                memory_extracted_through_message_id=conversation.messages[3].id,
                require_idle=True,
            ),
        )
    assert active.value.status_code == 409


def test_compaction_atomically_refreshes_core_memories_snapshot(repository):
    conversation = make_conversation(repository, count=4)
    refreshed = datetime(2026, 11, 26, 14, 0, tzinfo=UTC)

    updated = repository.update_conversation_compaction(
        "alpha",
        conversation.id,
        ConversationCompactionUpdate(
            expected_version=conversation.version,
            compacted_summary="Summary text.",
            compacted_through_message_id=conversation.messages[2].id,
            core_memories_snapshot="Core memories:\n- **Name**: Alice",
            core_memories_refreshed_at=refreshed,
        ),
    )

    assert updated.compacted_summary == "Summary text."
    assert updated.core_memories_snapshot == "Core memories:\n- **Name**: Alice"
    assert updated.core_memories_refreshed_at == refreshed


def test_manual_memory_edits_invalidate_snapshot_while_background_preserves_it(tmp_path):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        client.post("/api/agents", json={"key": "alpha", "name": "Alpha"})
        msg1 = str(uuid4())
        conv = client.post(
            "/api/agents/alpha/resources",
            json={
                "kind": "conversation",
                "title": "New chat",
                "messages": [
                    {
                        "id": msg1,
                        "role": "user",
                        "content": "Hello",
                        "created_at": "2026-11-26T10:00:00Z",
                    }
                ],
            },
        ).json()

        def set_snapshot(text: str):
            res = client.put(
                f"/api/agents/alpha/resources/{conv['id']}/memory-state",
                json={
                    "core_memories_snapshot": text,
                    "memory_extracted_through_message_id": msg1,
                },
            )
            assert res.status_code == 200
            return res.json()

        # Set initial snapshot.
        frozen = set_snapshot("Core memories:\n- **User**: Andrew")
        assert frozen["core_memories_snapshot"] == "Core memories:\n- **User**: Andrew"

        # Background memory creation with preserve_memory_snapshot=true keeps snapshot intact.
        bg_mem = client.post(
            "/api/agents/alpha/resources",
            params={"preserve_memory_snapshot": "true"},
            json={
                "kind": "memory",
                "title": "Favorite tea",
                "markdown": "Sencha green tea",
                "tags": ["core"],
            },
        ).json()
        reloaded = client.get(f"/api/agents/alpha/resources/{conv['id']}").json()
        assert reloaded["core_memories_snapshot"] == "Core memories:\n- **User**: Andrew"

        # Non-memory resource creation does not invalidate snapshot.
        client.post(
            "/api/agents/alpha/resources",
            json={"kind": "space", "title": "Doc", "markdown": "Notes"},
        )
        reloaded = client.get(f"/api/agents/alpha/resources/{conv['id']}").json()
        assert reloaded["core_memories_snapshot"] == "Core memories:\n- **User**: Andrew"

        # Manual memory update (default preserve_memory_snapshot=false) clears snapshot.
        updated_mem = client.put(
            f"/api/agents/alpha/resources/{bg_mem['id']}",
            json={
                "kind": "memory",
                "title": "Favorite tea",
                "markdown": "Matcha",
                "tags": ["core"],
                "expected_version": bg_mem["version"],
            },
        ).json()
        reloaded = client.get(f"/api/agents/alpha/resources/{conv['id']}").json()
        assert reloaded["core_memories_snapshot"] is None
        assert reloaded["core_memories_refreshed_at"] is None
        assert reloaded["memory_extracted_through_message_id"] == msg1

        # Re-freeze, then trash memory -> clears snapshot.
        set_snapshot("Core memories:\n- **Favorite tea**: Matcha")
        trashed = client.delete(
            f"/api/agents/alpha/resources/{updated_mem['id']}",
            params={"expected_version": updated_mem["version"]},
        ).json()
        reloaded = client.get(f"/api/agents/alpha/resources/{conv['id']}").json()
        assert reloaded["core_memories_snapshot"] is None

        # Re-freeze, then restore memory -> clears snapshot.
        set_snapshot("")
        client.post(
            f"/api/agents/alpha/resources/{trashed['id']}/restore",
            json={"expected_version": trashed["version"]},
        )
        reloaded = client.get(f"/api/agents/alpha/resources/{conv['id']}").json()
        assert reloaded["core_memories_snapshot"] is None

        # Metadata-only conversation PUT preserves cursor and snapshot.
        set_snapshot("Core memories:\n- **Favorite tea**: Matcha")
        before_rename = client.get(f"/api/agents/alpha/resources/{conv['id']}").json()
        renamed = client.put(
            f"/api/agents/alpha/resources/{conv['id']}",
            json={
                "kind": "conversation",
                "title": "Renamed",
                "messages": before_rename["messages"],
                "expected_version": before_rename["version"],
            },
        ).json()
        assert renamed["core_memories_snapshot"] == "Core memories:\n- **Favorite tea**: Matcha"
        assert renamed["memory_extracted_through_message_id"] == msg1


def test_memory_state_rejects_naive_timestamp():
    with pytest.raises(ValidationError):
        ConversationMemoryStateUpdate(
            core_memories_snapshot="Test",
            core_memories_refreshed_at=datetime(2026, 11, 26, 12, 0),
        )
