"""Unit tests for Group Chat durable storage, atomic turn-cap enforcement, and MCP Events."""

import hashlib
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from tsuchi.config import Settings
from tsuchi.main import create_app


def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@pytest.fixture
def client(tmp_path):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as test_client:
        yield test_client


def _seed_room_and_participants(client: TestClient, *, max_turns: int = 3):
    room = client.get("/api/group-chat/rooms/default").json()
    room = client.put(
        f"/api/group-chat/rooms/{room['id']}",
        json={
            "max_consecutive_agent_turns": max_turns,
            "expected_version": room["version"],
        },
    ).json()
    kadan = client.post(
        f"/api/group-chat/rooms/{room['id']}/participants",
        json={
            "display_name": "Kadan",
            "kind": "kadan_agent",
            "agent_key": "kadan",
            "description": "Built-in Kadan agent",
        },
    ).json()
    dot_token = "kdn_gc_dot_secret_token_1234567890"
    dot = client.post(
        f"/api/group-chat/rooms/{room['id']}/participants",
        json={
            "display_name": "Dot",
            "kind": "external_agent",
            "description": "External coding assistant",
            "token_hash": _hash_token(dot_token),
        },
    ).json()
    return room, kadan, dot, dot_token


def test_default_room_participants_handles_and_token_hashing(client):
    room = client.get("/api/group-chat/rooms/default")
    assert room.status_code == 200
    room_data = room.json()
    assert room_data["title"] == "Group Chat"
    assert room_data["consecutive_agent_turns"] == 0
    assert room_data["version"] == 1

    # Duplicate display names get unique derived handles.
    token_a = "kdn_gc_token_alpha_1234567890"
    token_b = "kdn_gc_token_beta_1234567890"
    p1 = client.post(
        f"/api/group-chat/rooms/{room_data['id']}/participants",
        json={
            "display_name": "Dot Agent",
            "kind": "external_agent",
            "token_hash": _hash_token(token_a),
        },
    )
    assert p1.status_code == 201
    p1_data = p1.json()
    assert p1_data["handle"] == "dot-agent"
    assert "token_hash" not in p1_data

    p2 = client.post(
        f"/api/group-chat/rooms/{room_data['id']}/participants",
        json={
            "display_name": "Dot Agent",
            "kind": "external_agent",
            "token_hash": _hash_token(token_b),
        },
    )
    assert p2.status_code == 201
    assert p2.json()["handle"] == "dot-agent-2"

    # Listed participants never leak token hashes.
    listed = client.get(f"/api/group-chat/rooms/{room_data['id']}/participants").json()
    assert len(listed) == 2
    assert all("token_hash" not in item for item in listed)

    # Authenticate with valid token updates last_seen_at.
    authed = client.post(
        "/api/group-chat/participants/authenticate",
        json={"token_hash": _hash_token(token_a)},
    )
    assert authed.status_code == 200
    assert authed.json()["id"] == p1_data["id"]
    assert authed.json()["last_seen_at"] is not None

    # Invalid token rejected with 401.
    bad = client.post(
        "/api/group-chat/participants/authenticate",
        json={"token_hash": _hash_token("wrong-token")},
    )
    assert bad.status_code == 401
    assert bad.json()["code"] == "invalid_token"

    # Pausing rejects auth with 403.
    paused = client.put(
        f"/api/group-chat/rooms/{room_data['id']}/participants/{p1_data['id']}",
        json={"enabled": False, "expected_version": p1_data["version"]},
    )
    assert paused.status_code == 200
    assert paused.json()["enabled"] is False
    paused_auth = client.post(
        "/api/group-chat/participants/authenticate",
        json={"token_hash": _hash_token(token_a)},
    )
    assert paused_auth.status_code == 403
    assert paused_auth.json()["code"] == "participant_paused"

    # Resuming and rotating token invalidates old token and accepts new token.
    new_token = "kdn_gc_token_alpha_rotated_9876543210"
    rotated = client.put(
        f"/api/group-chat/rooms/{room_data['id']}/participants/{p1_data['id']}",
        json={
            "enabled": True,
            "token_hash": _hash_token(new_token),
            "expected_version": paused.json()["version"],
        },
    )
    assert rotated.status_code == 200
    assert (
        client.post(
            "/api/group-chat/participants/authenticate",
            json={"token_hash": _hash_token(token_a)},
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/group-chat/participants/authenticate",
            json={"token_hash": _hash_token(new_token)},
        ).status_code
        == 200
    )

    # Revoking clears token and blocks further edits.
    revoked = client.put(
        f"/api/group-chat/rooms/{room_data['id']}/participants/{p1_data['id']}",
        json={"revoke": True, "expected_version": rotated.json()["version"]},
    )
    assert revoked.status_code == 200
    assert revoked.json()["revoked_at"] is not None
    assert revoked.json()["enabled"] is False
    assert (
        client.post(
            "/api/group-chat/participants/authenticate",
            json={"token_hash": _hash_token(new_token)},
        ).status_code
        == 401
    )


def test_monotonic_seq_pagination_and_client_request_id_idempotency(client):
    room, kadan, dot, _ = _seed_room_and_participants(client, max_turns=6)
    room_id = room["id"]

    # Post 3 messages; verify monotonic seq.
    r1 = client.post(
        f"/api/group-chat/rooms/{room_id}/messages",
        json={
            "sender_kind": "user",
            "sender_name": "You",
            "content": "Hey @kadan, start us off",
            "handoff_to_participant_ids": [kadan["id"]],
            "client_request_id": "req-user-1",
        },
    )
    assert r1.status_code == 201
    assert r1.json()["message"]["seq"] == 1
    assert r1.json()["idempotent_replay"] is False

    # Replay of same client_request_id returns existing message without new seq.
    r1_dup = client.post(
        f"/api/group-chat/rooms/{room_id}/messages",
        json={
            "sender_kind": "user",
            "sender_name": "You",
            "content": "Hey @kadan, start us off",
            "handoff_to_participant_ids": [kadan["id"]],
            "client_request_id": "req-user-1",
        },
    )
    assert r1_dup.status_code == 201
    assert r1_dup.json()["idempotent_replay"] is True
    assert r1_dup.json()["message"]["id"] == r1.json()["message"]["id"]
    assert r1_dup.json()["message"]["seq"] == 1

    # Conflicting payload with same client_request_id returns 409.
    conflict = client.post(
        f"/api/group-chat/rooms/{room_id}/messages",
        json={
            "sender_kind": "user",
            "sender_name": "You",
            "content": "Different content",
            "client_request_id": "req-user-1",
        },
    )
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "idempotency_conflict"

    r2 = client.post(
        f"/api/group-chat/rooms/{room_id}/messages",
        json={
            "sender_kind": "kadan_agent",
            "participant_id": kadan["id"],
            "sender_name": "Kadan",
            "content": "Passing to @dot",
            "handoff_to_participant_ids": [dot["id"]],
            "client_request_id": "req-kadan-1",
        },
    )
    assert r2.status_code == 201
    assert r2.json()["message"]["seq"] == 2

    r3 = client.post(
        f"/api/group-chat/rooms/{room_id}/messages",
        json={
            "sender_kind": "external_agent",
            "participant_id": dot["id"],
            "sender_name": "Dot",
            "content": "Got it!",
            "client_request_id": "req-dot-1",
        },
    )
    assert r3.status_code == 201
    assert r3.json()["message"]["seq"] == 3

    # Self-handoff is rejected.
    self_handoff = client.post(
        f"/api/group-chat/rooms/{room_id}/messages",
        json={
            "sender_kind": "external_agent",
            "participant_id": dot["id"],
            "sender_name": "Dot",
            "content": "Talking to myself",
            "handoff_to_participant_ids": [dot["id"]],
        },
    )
    assert self_handoff.status_code == 422
    assert self_handoff.json()["code"] == "self_handoff_not_allowed"

    # Pagination via after_seq and before_seq.
    after_1 = client.get(
        f"/api/group-chat/rooms/{room_id}/messages", params={"after_seq": 1, "limit": 1}
    ).json()
    assert [m["seq"] for m in after_1["messages"]] == [2]
    assert after_1["has_more"] is True
    assert after_1["latest_seq"] == 3

    before_3 = client.get(
        f"/api/group-chat/rooms/{room_id}/messages",
        params={"before_seq": 3, "limit": 10},
    ).json()
    assert [m["seq"] for m in before_3["messages"]] == [1, 2]
    assert before_3["has_more"] is False


def test_atomic_turn_cap_enforcement_suppression_and_user_reset(client):
    room, kadan, dot, _ = _seed_room_and_participants(client, max_turns=2)
    room_id = room["id"]

    # Register an active webhook subscription for Dot so we verify webhook suppression at cap.
    sub = client.post(
        f"/api/group-chat/rooms/{room_id}/subscriptions",
        json={
            "participant_id": dot["id"],
            "event_name": "group_chat.turn_requested",
            "callback_url": "https://dot.example/webhook",
            "signing_secret": "whsec_dGVzdC1zZWNyZXQtMTIzNDU2Nzg5MDEyMzQ1Ng==",
            "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        },
    )
    assert sub.status_code == 201
    sub_id = sub.json()["id"]

    # Turn 1 (kadan -> dot): accepted, turns=1 < 2, enqueues webhook delivery.
    t1 = client.post(
        f"/api/group-chat/rooms/{room_id}/messages",
        json={
            "sender_kind": "kadan_agent",
            "participant_id": kadan["id"],
            "sender_name": "Kadan",
            "content": "Over to @dot",
            "handoff_to_participant_ids": [dot["id"]],
            "client_request_id": "turn-1",
        },
    )
    assert t1.status_code == 201
    t1_data = t1.json()
    assert t1_data["room"]["consecutive_agent_turns"] == 1
    assert t1_data["turn_cap_reached"] is False
    assert t1_data["handoff_suppressed"] is False
    assert t1_data["effective_handoff_to_participant_ids"] == [dot["id"]]
    assert len(t1_data["enqueued_delivery_ids"]) == 1

    # Turn 2 (dot -> kadan, also tagging dot not allowed so tag kadan): hits cap (2 == 2)!
    # Handoff MUST be suppressed and turn_cap_reached == True.
    t2 = client.post(
        f"/api/group-chat/rooms/{room_id}/messages",
        json={
            "sender_kind": "external_agent",
            "participant_id": dot["id"],
            "sender_name": "Dot",
            "content": "Back to @kadan",
            "handoff_to_participant_ids": [kadan["id"]],
            "client_request_id": "turn-2",
        },
    )
    assert t2.status_code == 201
    t2_data = t2.json()
    assert t2_data["room"]["consecutive_agent_turns"] == 2
    assert t2_data["turn_cap_reached"] is True
    assert t2_data["handoff_suppressed"] is True
    assert t2_data["effective_handoff_to_participant_ids"] == []
    assert t2_data["enqueued_delivery_ids"] == []

    # Idempotent retry of turn 2 succeeds without triggering agent_turn_cap_reached!
    t2_retry = client.post(
        f"/api/group-chat/rooms/{room_id}/messages",
        json={
            "sender_kind": "external_agent",
            "participant_id": dot["id"],
            "sender_name": "Dot",
            "content": "Back to @kadan",
            "handoff_to_participant_ids": [kadan["id"]],
            "client_request_id": "turn-2",
        },
    )
    assert t2_retry.status_code == 201
    assert t2_retry.json()["idempotent_replay"] is True
    assert t2_retry.json()["message"]["id"] == t2_data["message"]["id"]

    # Subsequent send from Kadan agent rejected with agent_turn_cap_reached.
    t3_kadan = client.post(
        f"/api/group-chat/rooms/{room_id}/messages",
        json={
            "sender_kind": "kadan_agent",
            "participant_id": kadan["id"],
            "sender_name": "Kadan",
            "content": "Trying past cap",
            "client_request_id": "turn-3-kadan",
        },
    )
    assert t3_kadan.status_code == 409
    assert t3_kadan.json()["code"] == "agent_turn_cap_reached"

    # Subsequent send from external agent also rejected with agent_turn_cap_reached.
    t3_dot = client.post(
        f"/api/group-chat/rooms/{room_id}/messages",
        json={
            "sender_kind": "external_agent",
            "participant_id": dot["id"],
            "sender_name": "Dot",
            "content": "Trying past cap too",
            "client_request_id": "turn-3-dot",
        },
    )
    assert t3_dot.status_code == 409
    assert t3_dot.json()["code"] == "agent_turn_cap_reached"

    # User message resets consecutive_agent_turns to 0 atomically and re-enables handoffs/webhooks.
    user_reset = client.post(
        f"/api/group-chat/rooms/{room_id}/messages",
        json={
            "sender_kind": "user",
            "sender_name": "You",
            "content": "Keep going @dot",
            "handoff_to_participant_ids": [dot["id"]],
        },
    )
    assert user_reset.status_code == 201
    assert user_reset.json()["room"]["consecutive_agent_turns"] == 0
    assert user_reset.json()["turn_cap_reached"] is False
    assert len(user_reset.json()["enqueued_delivery_ids"]) == 1

    # Verify webhook deliveries exist for seq 1 and seq 3 (NOT seq 2), and delivery updates advance last_delivered_seq.
    deliveries = client.get(
        "/api/group-chat/webhook-deliveries", params={"status": "pending"}
    ).json()
    assert [d["seq"] for d in deliveries] == [1, 3]
    first_delivery = deliveries[0]
    done = client.put(
        f"/api/group-chat/webhook-deliveries/{first_delivery['id']}",
        json={
            "status": "delivered",
            "attempts": 1,
            "expected_version": first_delivery["version"],
        },
    )
    assert done.status_code == 200
    sub_after = client.get(f"/api/group-chat/subscriptions/{sub_id}").json()
    assert sub_after["last_delivered_seq"] == 1

    # Revoking participant immediately revokes active subscription and drops pending deliveries.
    revoked = client.put(
        f"/api/group-chat/rooms/{room_id}/participants/{dot['id']}",
        json={"revoke": True, "expected_version": dot["version"]},
    )
    assert revoked.status_code == 200
    sub_revoked = client.get(f"/api/group-chat/subscriptions/{sub_id}").json()
    assert sub_revoked["status"] == "revoked"
    pending_after = client.get(
        "/api/group-chat/webhook-deliveries", params={"status": "pending"}
    ).json()
    assert pending_after == []


def test_group_chat_event_run_allows_omitted_origin_run_id(client):
    assert (
        client.post("/api/agents", json={"key": "kadan", "name": "Kadan"}).status_code
        == 201
    )
    persona = client.put(
        "/api/agents/kadan/persona",
        json={
            "title": "Kadan",
            "markdown": "Friendly teammate",
            "expected_version": None,
        },
    ).json()
    conv = client.post(
        "/api/agents/kadan/resources",
        json={"kind": "conversation", "title": "Chat"},
    ).json()

    run_id = str(uuid4())
    created = client.post(
        f"/api/agents/kadan/resources/{conv['id']}/event-runs",
        json={
            "id": run_id,
            "event": {
                "kind": "group_chat",
                "content": '{"room_id":"00000000-0000-0000-0000-000000000001","trigger_seq":1}',
            },
        },
    )
    assert created.status_code == 201
    run = created.json()
    assert run["id"] == run_id
    assert run["persona_id"] == persona["id"]
    assert run["references"] == []
    assert run["trigger_event"]["kind"] == "group_chat"

    # Idempotent replay with same id and event returns same run.
    replay = client.post(
        f"/api/agents/kadan/resources/{conv['id']}/event-runs",
        json={
            "id": run_id,
            "event": {
                "kind": "group_chat",
                "content": '{"room_id":"00000000-0000-0000-0000-000000000001","trigger_seq":1}',
            },
        },
    )
    assert replay.status_code == 201
    assert replay.json()["id"] == run_id
