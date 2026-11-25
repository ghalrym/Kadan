"""Unit tests for global MCP server storage, OAuth states, and T3 thread watches."""

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


def test_mcp_server_crud_and_version_conflict(client):
    created = client.post(
        "/api/mcp-servers",
        json={
            "slug": "t3-code",
            "name": "T3 Code",
            "preset": "t3_code",
            "url": "http://127.0.0.1:3773/mcp",
            "auth_type": "oauth",
            "pinned_tools": ["t3_thread_launch", "t3_thread_read"],
        },
    )
    assert created.status_code == 201
    server = created.json()
    assert server["slug"] == "t3-code"
    assert server["version"] == 1
    assert server["status"] == "disconnected"

    duplicate = client.post(
        "/api/mcp-servers",
        json={
            "slug": "t3-code",
            "name": "Duplicate",
            "url": "http://127.0.0.1:3773/mcp",
        },
    )
    assert duplicate.status_code == 409

    listed = client.get("/api/mcp-servers")
    assert listed.status_code == 200
    assert [row["id"] for row in listed.json()] == [server["id"]]

    updated = client.put(
        f"/api/mcp-servers/{server['id']}",
        json={
            "slug": "t3-code",
            "name": "T3 Code",
            "preset": "t3_code",
            "url": "http://127.0.0.1:3773/mcp",
            "auth_type": "oauth",
            "status": "connected",
            "pinned_tools": ["t3_thread_launch"],
            "tools": [
                {
                    "name": "t3_thread_launch",
                    "description": "Create a thread",
                    "input_schema": {
                        "type": "object",
                        "properties": {"clientRequestId": {"type": "string"}},
                    },
                    "read_only": False,
                    "idempotent": True,
                    "accepts_client_request_id": True,
                }
            ],
            "expected_version": 1,
        },
    )
    assert updated.status_code == 200
    assert updated.json()["version"] == 2
    assert updated.json()["status"] == "connected"

    stale = client.put(
        f"/api/mcp-servers/{server['id']}",
        json={
            "slug": "t3-code",
            "name": "Stale",
            "url": "http://127.0.0.1:3773/mcp",
            "expected_version": 1,
        },
    )
    assert stale.status_code == 409

    deleted = client.delete(f"/api/mcp-servers/{server['id']}")
    assert deleted.status_code == 204
    assert client.get("/api/mcp-servers").json() == []


def test_mcp_oauth_state_is_single_use_and_expires(client):
    server_id = str(uuid4())
    future = (datetime.now(UTC) + timedelta(minutes=10)).isoformat()
    saved = client.post(
        "/api/mcp-oauth-states",
        json={
            "state": "state-" + "a" * 24,
            "server_id": server_id,
            "code_verifier": "verifier-" + "b" * 40,
            "redirect_uri": "http://localhost:8630/api/mcp/oauth/callback",
            "token_endpoint": "http://localhost:3773/oauth/token",
            "client_id": "kadan-client",
            "expires_at": future,
        },
    )
    assert saved.status_code == 201
    state_key = saved.json()["state"]

    consumed = client.post(f"/api/mcp-oauth-states/{state_key}/consume")
    assert consumed.status_code == 200
    assert consumed.json()["server_id"] == server_id

    again = client.post(f"/api/mcp-oauth-states/{state_key}/consume")
    assert again.status_code == 404

    past = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    expired = client.post(
        "/api/mcp-oauth-states",
        json={
            "state": "state-" + "c" * 24,
            "server_id": server_id,
            "code_verifier": "verifier-" + "d" * 40,
            "redirect_uri": "http://localhost:8630/api/mcp/oauth/callback",
            "token_endpoint": "http://localhost:3773/oauth/token",
            "client_id": "kadan-client",
            "expires_at": past,
        },
    )
    assert expired.status_code == 201
    assert (
        client.post(f"/api/mcp-oauth-states/{expired.json()['state']}/consume").status_code
        == 410
    )


def test_t3_watches_are_idempotent_and_updatable(client):
    watch_id = str(uuid4())
    payload = {
        "id": watch_id,
        "server_id": str(uuid4()),
        "thread_id": "thread-123",
        "run_id": "run-1",
        "thread_link": "https://t3.local/threads/thread-123",
        "thread_title": "Refactor MCP",
        "origin_agent_key": "mi",
        "origin_conversation_id": str(uuid4()),
        "origin_run_id": str(uuid4()),
    }
    first = client.post("/api/t3-watches", json=payload)
    assert first.status_code == 201
    second = client.post("/api/t3-watches", json=payload)
    assert second.status_code == 201
    assert second.json()["id"] == watch_id
    assert second.json()["version"] == 1

    watching = client.get("/api/t3-watches", params={"status": "watching"}).json()
    assert len(watching) == 1

    updated = client.put(
        f"/api/t3-watches/{watch_id}",
        json={
            "status": "watching",
            "last_pending_request_id": "req-1",
            "expected_version": 1,
        },
    )
    assert updated.status_code == 200
    assert updated.json()["last_pending_request_id"] == "req-1"
    assert updated.json()["version"] == 2

    done = client.put(
        f"/api/t3-watches/{watch_id}",
        json={
            "status": "delivered",
            "last_pending_request_id": "req-1",
            "expected_version": 2,
        },
    )
    assert done.status_code == 200
    assert client.get("/api/t3-watches", params={"status": "watching"}).json() == []
