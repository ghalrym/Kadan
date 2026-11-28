"""Unit tests for direct (1-on-1) Group Chat rooms between the user and one external agent."""

import hashlib
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from tsuchi.config import Settings
from tsuchi.main import create_app
from tsuchi.repository import Repository


@pytest.fixture
def client(tmp_path):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as test_client:
        yield test_client


def _participants(client: TestClient):
    room = client.get("/api/group-chat/rooms/default").json()
    kadan = client.post(
        f"/api/group-chat/rooms/{room['id']}/participants",
        json={"display_name": "Kadan", "kind": "kadan_agent", "agent_key": "kadan"},
    ).json()
    dot = client.post(
        f"/api/group-chat/rooms/{room['id']}/participants",
        json={
            "display_name": "Dot",
            "kind": "external_agent",
            "token_hash": hashlib.sha256(b"dot-token-1234567890").hexdigest(),
        },
    ).json()
    sol = client.post(
        f"/api/group-chat/rooms/{room['id']}/participants",
        json={
            "display_name": "Sol",
            "kind": "external_agent",
            "token_hash": hashlib.sha256(b"sol-token-1234567890").hexdigest(),
        },
    ).json()
    return room, kadan, dot, sol


def _message(client, room_id, **body):
    return client.post(f"/api/group-chat/rooms/{room_id}/messages", json=body)


def test_direct_room_is_stable_private_and_only_for_external_agents(client):
    room, kadan, dot, sol = _participants(client)

    first = client.post(f"/api/group-chat/participants/{dot['id']}/direct-room")
    assert first.status_code == 200
    direct = first.json()
    assert direct["kind"] == "direct"
    assert direct["direct_participant_id"] == dot["id"]
    assert direct["title"] == "Dot"
    assert client.post(f"/api/group-chat/participants/{dot['id']}/direct-room").json() == direct
    assert direct["id"] != room["id"]

    assert client.post(f"/api/group-chat/participants/{kadan['id']}/direct-room").status_code == 409

    # Dot can post in its own direct room; Sol and Kadan's agent cannot.
    assert _message(client, direct["id"], sender_kind="user", sender_name="You", content="Hi Dot",
                    handoff_to_participant_ids=[dot["id"]]).status_code == 201
    assert _message(client, direct["id"], sender_kind="external_agent", participant_id=dot["id"],
                    sender_name="Dot", content="Hi!").status_code == 201
    assert _message(client, direct["id"], sender_kind="external_agent", participant_id=sol["id"],
                    sender_name="Sol", content="Let me in").status_code == 404
    assert _message(client, direct["id"], sender_kind="kadan_agent", participant_id=kadan["id"],
                    sender_name="Kadan", content="Peek").status_code == 404
    assert _message(client, direct["id"], sender_kind="user", sender_name="You", content="Tag Sol",
                    handoff_to_participant_ids=[sol["id"]]).status_code == 404

    # Group Chat history is untouched by the direct conversation.
    group_messages = client.get(f"/api/group-chat/rooms/{room['id']}/messages").json()["messages"]
    assert group_messages == []

    summaries = client.get("/api/group-chat/direct-rooms").json()
    assert [(item["room"]["id"], item["latest_seq"]) for item in summaries] == [(direct["id"], 2)]
    assert summaries[0]["last_message_at"]


def test_direct_messages_wake_the_agent_through_its_group_subscription(client):
    room, _kadan, dot, _ = _participants(client)
    sub = client.post(
        f"/api/group-chat/rooms/{room['id']}/subscriptions",
        json={
            "participant_id": dot["id"],
            "event_name": "group_chat.turn_requested",
            "filters": {"room_id": room["id"]},
            "callback_url": "https://dot.example/webhook",
            "signing_secret": "whsec_dGVzdC1zZWNyZXQtMTIzNDU2Nzg5MDEyMzQ1Ng==",
            "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        },
    )
    assert sub.status_code == 201
    direct = client.post(f"/api/group-chat/participants/{dot['id']}/direct-room").json()

    # Both rooms start at seq 1; the same subscription must get both deliveries.
    group = _message(client, room["id"], sender_kind="user", sender_name="You", content="@dot group",
                     handoff_to_participant_ids=[dot["id"]])
    private = _message(client, direct["id"], sender_kind="user", sender_name="You", content="private",
                       handoff_to_participant_ids=[dot["id"]])
    assert group.json()["message"]["seq"] == private.json()["message"]["seq"] == 1
    assert len(group.json()["enqueued_delivery_ids"]) == 1
    assert len(private.json()["enqueued_delivery_ids"]) == 1

    deliveries = client.get("/api/group-chat/webhook-deliveries", params={"status": "pending"}).json()
    by_room = {item["room_id"]: item for item in deliveries}
    assert by_room[direct["id"]]["payload"]["room_kind"] == "direct"
    assert by_room[direct["id"]]["payload"]["message"]["content"] == "private"
    assert by_room[room["id"]]["payload"]["room_kind"] == "group"


def test_old_delivery_uniqueness_is_migrated_without_losing_rows(tmp_path):
    repository = Repository(tmp_path)
    repository.initialize()
    with sqlite3.connect(repository.database_path) as db:
        db.execute("DROP TABLE group_chat_webhook_deliveries")
        db.execute(
            """CREATE TABLE group_chat_webhook_deliveries (
                id TEXT PRIMARY KEY, subscription_id TEXT NOT NULL, room_id TEXT NOT NULL,
                participant_id TEXT NOT NULL, seq INTEGER NOT NULL, status TEXT NOT NULL,
                next_attempt_at TEXT NOT NULL, updated_at TEXT NOT NULL, data TEXT NOT NULL,
                UNIQUE(subscription_id, seq))"""
        )
        db.execute(
            "INSERT INTO group_chat_webhook_deliveries VALUES ('d1', 's1', 'r1', 'p1', 1, 'pending', 't', 't', '{}')"
        )

    repository.initialize()

    with sqlite3.connect(repository.database_path) as db:
        sql = db.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'group_chat_webhook_deliveries'"
        ).fetchone()[0]
        assert "UNIQUE(subscription_id, room_id, seq)" in sql
        assert db.execute("SELECT id FROM group_chat_webhook_deliveries").fetchall() == [("d1",)]
        db.execute(
            "INSERT INTO group_chat_webhook_deliveries VALUES ('d2', 's1', 'r2', 'p1', 1, 'pending', 't', 't', '{}')"
        )
