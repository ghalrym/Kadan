"""Unit tests for native Work durable storage: projects, threads, buckets, snooze, runs, activity."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from tsuchi.config import Settings
from tsuchi.main import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as test_client:
        yield test_client


def test_work_project_crud_and_delete_guard(client):
    created = client.post(
        "/api/work/projects",
        json={
            "title": "Kadan",
            "root_path": "/srv/projects/kadan",
        },
    )
    assert created.status_code == 201
    project = created.json()
    assert project["version"] == 1
    assert project["root_path"] == "/srv/projects/kadan"

    # Duplicate root_path is rejected.
    dup = client.post(
        "/api/work/projects",
        json={"title": "Dup", "root_path": "/srv/projects/kadan"},
    )
    assert dup.status_code == 409

    # Optimistic-locking update.
    updated = client.put(
        f"/api/work/projects/{project['id']}",
        json={
            "title": "Kadan Repo",
            "root_path": "/srv/projects/kadan",
            "expected_version": 1,
        },
    )
    assert updated.status_code == 200
    assert updated.json()["version"] == 2
    assert updated.json()["title"] == "Kadan Repo"

    stale = client.put(
        f"/api/work/projects/{project['id']}",
        json={
            "title": "Stale",
            "root_path": "/srv/projects/kadan",
            "expected_version": 1,
        },
    )
    assert stale.status_code == 409

    # Create a thread referencing the project; non-force delete is rejected.
    thread = client.post(
        "/api/work/threads",
        json={
            "project_id": project["id"],
            "title": "Fix login",
            "harness": "codex",
        },
    )
    assert thread.status_code == 201
    assert client.delete(f"/api/work/projects/{project['id']}").status_code == 409

    # Force delete removes project and cascading threads.
    assert (
        client.delete(
            f"/api/work/projects/{project['id']}", params={"force": "true"}
        ).status_code
        == 204
    )
    assert client.get("/api/work/projects").json() == []
    assert client.get("/api/work/threads").json() == []


def test_work_thread_idempotency_buckets_and_snooze_wakeup(client):
    project = client.post(
        "/api/work/projects",
        json={"title": "Demo", "root_path": "/tmp/demo"},
    ).json()

    payload = {
        "client_request_id": "launch-action-1",
        "project_id": project["id"],
        "title": "Add sidebar",
        "harness": "claude",
        "model": "claude-sonnet-4-6",
        "origin_agent_key": "kadan",
        "origin_conversation_id": str(uuid4()),
        "origin_run_id": str(uuid4()),
    }
    first = client.post("/api/work/threads", json=payload)
    assert first.status_code == 201
    second = client.post("/api/work/threads", json=payload)
    assert second.status_code == 201
    assert first.json()["id"] == second.json()["id"]
    thread_id = first.json()["id"]

    # Snooze without snoozed_until is rejected.
    bad_snooze = client.post(
        f"/api/work/threads/{thread_id}/organize",
        json={"bucket": "snoozed"},
    )
    assert bad_snooze.status_code == 422

    # Snooze into the past, then wake via wake-snoozed.
    past = (datetime.now(UTC) - timedelta(minutes=2)).isoformat()
    snoozed = client.post(
        f"/api/work/threads/{thread_id}/organize",
        json={"bucket": "snoozed", "snoozed_until": past, "expected_version": 1},
    )
    assert snoozed.status_code == 200
    assert snoozed.json()["bucket"] == "snoozed"
    assert snoozed.json()["version"] == 2

    # Future-snoozed thread stays snoozed while past-snoozed thread wakes.
    future = (datetime.now(UTC) + timedelta(hours=2)).isoformat()
    other = client.post(
        "/api/work/threads",
        json={
            "project_id": project["id"],
            "title": "Later task",
            "harness": "grok",
            "bucket": "snoozed",
            "snoozed_until": future,
        },
    ).json()

    woken = client.post("/api/work/threads/wake-snoozed")
    assert woken.status_code == 200
    woken_ids = [item["id"] for item in woken.json()]
    assert woken_ids == [thread_id]
    assert woken.json()[0]["bucket"] == "default"
    assert woken.json()[0]["snoozed_until"] is None

    still_snoozed = client.get(f"/api/work/threads/{other['id']}").json()
    assert still_snoozed["bucket"] == "snoozed"

    # Settle and unsettle.
    settled = client.post(
        f"/api/work/threads/{thread_id}/organize",
        json={"bucket": "settled"},
    )
    assert settled.status_code == 200
    assert settled.json()["bucket"] == "settled"

    by_settled = client.get(
        "/api/work/threads", params={"bucket": "settled", "q": "sidebar"}
    ).json()
    assert [row["id"] for row in by_settled] == [thread_id]

    unsettled = client.post(
        f"/api/work/threads/{thread_id}/organize",
        json={"bucket": "default"},
    )
    assert unsettled.status_code == 200
    assert unsettled.json()["bucket"] == "default"


def test_work_runs_activity_and_pending_requests(client):
    project = client.post(
        "/api/work/projects",
        json={"title": "App", "root_path": "/tmp/app"},
    ).json()
    thread = client.post(
        "/api/work/threads",
        json={
            "project_id": project["id"],
            "title": "Build feature",
            "harness": "antigravity",
        },
    ).json()
    thread_id = thread["id"]

    # Idempotent run creation moves thread to working.
    run1 = client.post(
        f"/api/work/threads/{thread_id}/runs",
        json={"client_request_id": "send-1", "prompt": "Implement feature"},
    )
    assert run1.status_code == 201
    run1_replay = client.post(
        f"/api/work/threads/{thread_id}/runs",
        json={"client_request_id": "send-1", "prompt": "Implement feature"},
    )
    assert run1_replay.status_code == 201
    assert run1.json()["id"] == run1_replay.json()["id"]
    assert run1.json()["ordinal"] == 1

    reloaded = client.get(f"/api/work/threads/{thread_id}").json()
    assert reloaded["bucket"] == "working"
    assert reloaded["run_status"] == "queued"

    # Claim run with lease & owner.
    owner = str(uuid4())
    lease = (datetime.now(UTC) + timedelta(seconds=60)).isoformat()
    claimed = client.put(
        f"/api/work/runs/{run1.json()['id']}",
        json={
            "status": "running",
            "owner_id": owner,
            "lease_expires_at": lease,
            "expected_version": 1,
        },
    )
    assert claimed.status_code == 200
    assert claimed.json()["version"] == 2

    # Pending approval moves thread to default/waiting so user notices it.
    paused = client.post(
        f"/api/work/threads/{thread_id}/pending-requests",
        json={
            "id": "perm-1",
            "kind": "approval",
            "title": "Run pytest",
            "prompt": "Allow running pytest?",
            "tool_name": "bash",
            "arguments": {"command": "pytest"},
        },
    )
    assert paused.status_code == 200
    assert paused.json()["run_status"] == "waiting"
    assert paused.json()["bucket"] == "default"
    assert len(paused.json()["pending_requests"]) == 1

    # Responding is idempotent and moves thread back to working.
    answered = client.post(
        f"/api/work/threads/{thread_id}/pending-requests/perm-1/respond",
        json={"response": {"decision": "approve"}},
    )
    assert answered.status_code == 200
    assert answered.json()["run_status"] == "running"
    assert answered.json()["bucket"] == "working"
    assert answered.json()["pending_requests"][0]["status"] == "answered"

    answered_again = client.post(
        f"/api/work/threads/{thread_id}/pending-requests/perm-1/respond",
        json={"response": {"decision": "approve"}},
    )
    assert answered_again.status_code == 200
    assert answered_again.json()["version"] == answered.json()["version"]

    # Append idempotent activity items.
    act = client.post(
        f"/api/work/threads/{thread_id}/activity",
        json=[
            {
                "id": "item-1",
                "run_id": run1.json()["id"],
                "kind": "user_message",
                "text": "Implement feature",
            },
            {
                "id": "item-2",
                "run_id": run1.json()["id"],
                "kind": "assistant_message",
                "text": "Done!",
            },
        ],
    )
    assert act.status_code == 201
    assert [row["position"] for row in act.json()] == [0, 1]

    # Re-appending item-1 returns existing without duplicating.
    replay_act = client.post(
        f"/api/work/threads/{thread_id}/activity",
        json=[{"id": "item-1", "kind": "user_message", "text": "Implement feature"}],
    )
    assert replay_act.status_code == 201
    assert replay_act.json()[0]["position"] == 0
    assert len(client.get(f"/api/work/threads/{thread_id}/activity").json()) == 2
    assert (
        len(
            client.get(
                f"/api/work/threads/{thread_id}/activity",
                params={"after_position": 0},
            ).json()
        )
        == 1
    )

    # Cancel active run.
    cancelled = client.post(f"/api/work/runs/{run1.json()['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert client.get(f"/api/work/threads/{thread_id}").json()["bucket"] == "default"


def test_work_harness_config_and_work_event_run_kind(client):
    saved = client.put(
        "/api/work/harnesses/grok",
        json={
            "credential": "xai-secret",
            "auth_mode": "api_key",
            "account_label": "Grok",
        },
    )
    assert saved.status_code == 200
    assert saved.json()["version"] == 1
    assert client.get("/api/work/harnesses/grok").json()["account_label"] == "Grok"
    assert client.delete("/api/work/harnesses/grok").status_code == 204
    assert client.get("/api/work/harnesses/grok").json() is None

    # Verify RunEvent accepts kind="work".
    client.post("/api/agents", json={"key": "kadan", "name": "Kadan"})
    convo = client.post(
        "/api/agents/kadan/resources",
        json={"kind": "conversation", "title": "Chat"},
    ).json()
    msg_id = str(uuid4())
    origin_run = client.post(
        f"/api/agents/kadan/resources/{convo['id']}/runs",
        json={
            "expected_version": 1,
            "message": {
                "id": msg_id,
                "role": "user",
                "content": "Start work",
                "created_at": datetime.now(UTC).isoformat(),
            },
        },
    ).json()
    event_id = str(uuid4())
    event_run = client.post(
        f"/api/agents/kadan/resources/{convo['id']}/event-runs",
        json={
            "id": event_id,
            "origin_run_id": origin_run["id"],
            "event": {"kind": "work", "content": '{"event_type":"thread_settled"}'},
        },
    )
    assert event_run.status_code == 201
    assert event_run.json()["trigger_event"]["kind"] == "work"
