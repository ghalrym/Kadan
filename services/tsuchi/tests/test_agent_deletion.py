"""Unit tests for permanent agent deletion in Tsuchi."""

import io
import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock
from uuid import uuid4, uuid5

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tsuchi.config import Settings, StorageError
from tsuchi.main import create_app
from tsuchi.models import PET_ANIMATIONS
from tsuchi.services import FileStorage


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=(12, 34, 56)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client(tmp_path):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as test_client:
        yield test_client, tmp_path


def _seed_agent(client: TestClient, key: str, name: str, worker_id: str) -> dict:
    created = client.post("/api/agents", json={"key": key, "name": name})
    assert created.status_code == 201

    assigned = client.post(
        "/api/runners/runner-1/assign", json={"worker_id": worker_id}
    )
    assert assigned.status_code == 200
    assert key in assigned.json()["agent_keys"]

    # Upload an image asset (for pet atlas + deck slide) and a text file asset.
    atlas_upload = client.post(
        f"/api/agents/{key}/assets",
        files={"file": ("atlas.png", png_bytes(), "image/png")},
        data={"title": f"{name} Atlas"},
    )
    assert atlas_upload.status_code == 201
    atlas_id = atlas_upload.json()["id"]

    note_upload = client.post(
        f"/api/agents/{key}/assets",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
        data={"title": f"{name} Notes"},
    )
    assert note_upload.status_code == 201
    note_asset_id = note_upload.json()["id"]

    # Owned persona (sets agents.persona_id).
    persona = client.put(
        f"/api/agents/{key}/persona",
        json={
            "title": f"{name} Persona",
            "markdown": f"You are {name}.",
            "expected_version": None,
        },
    )
    assert persona.status_code == 200

    # Imported ready pet (sets agents.selected_pet_id) and a queued generated pet job.
    imported_pet_id = str(uuid4())
    imported_pet = client.post(
        f"/api/agents/{key}/pets/imports",
        json={
            "id": imported_pet_id,
            "prompt": f"{name} imported pet",
            "manifest": {
                "atlas_asset_id": atlas_id,
                "animations": [
                    {"name": anim, "row": row, "durations": [120]}
                    for row, anim in enumerate(PET_ANIMATIONS)
                ],
            },
        },
    )
    assert imported_pet.status_code == 201
    selected = client.put(
        f"/api/agents/{key}/pet", json={"pet_id": imported_pet_id}
    )
    assert selected.status_code == 200

    generated_pet = client.post(
        f"/api/agents/{key}/pets",
        json={"id": str(uuid4()), "prompt": f"{name} generated pet"},
    )
    assert generated_pet.status_code == 201

    # Folder, Space page, Deck, and Memory with two revisions.
    folder = client.post(
        f"/api/agents/{key}/resources",
        json={"kind": "folder", "title": f"{name} Folder"},
    )
    assert folder.status_code == 201
    folder_id = folder.json()["id"]

    space_page = client.post(
        f"/api/agents/{key}/resources",
        json={
            "kind": "space",
            "title": f"{name} Page",
            "markdown": "Space content",
            "parent_id": folder_id,
            "attachment_ids": [note_asset_id],
        },
    )
    assert space_page.status_code == 201

    deck = client.post(
        f"/api/agents/{key}/resources",
        json={
            "kind": "deck",
            "title": f"{name} Deck",
            "parent_id": folder_id,
            "deck": {
                "slides": [
                    {
                        "id": str(uuid4()),
                        "layout": "image",
                        "title": "Slide 1",
                        "image_asset_id": atlas_id,
                    }
                ]
            },
        },
    )
    assert deck.status_code == 201

    memory = client.post(
        f"/api/agents/{key}/resources",
        json={
            "kind": "memory",
            "title": f"{name} Memory",
            "markdown": "Initial fact",
        },
    )
    assert memory.status_code == 201
    memory_updated = client.put(
        f"/api/agents/{key}/resources/{memory.json()['id']}",
        json={
            "kind": "memory",
            "title": f"{name} Memory",
            "markdown": "Updated fact",
            "expected_version": 1,
        },
    )
    assert memory_updated.status_code == 200

    # Conversation, running AgentRun with ephemeral draft, and ImageJob.
    conversation = client.post(
        f"/api/agents/{key}/resources",
        json={"kind": "conversation", "title": "New chat"},
    )
    assert conversation.status_code == 201
    conversation_id = conversation.json()["id"]

    message_id = str(uuid4())
    run = client.post(
        f"/api/agents/{key}/resources/{conversation_id}/runs",
        json={
            "expected_version": 1,
            "message": {
                "id": message_id,
                "role": "user",
                "content": f"Hello {name}",
                "created_at": datetime.now(UTC).isoformat(),
            },
        },
    )
    assert run.status_code == 201
    run_id = run.json()["id"]

    claimed = client.put(
        f"/api/agents/{key}/runs/{run_id}",
        json={
            "transcript_format": 2,
            "expected_version": 1,
            "worker_id": worker_id,
            "status": "running",
            "owner_id": worker_id,
            "lease_expires_at": (datetime.now(UTC) + timedelta(minutes=2)).isoformat(),
        },
    )
    assert claimed.status_code == 200

    draft = client.put(
        f"/api/agents/{key}/runs/{run_id}/draft",
        json={"worker_id": worker_id, "content": f"Typing as {name}..."},
    )
    assert draft.status_code == 204

    image_job = client.post(
        f"/api/agents/{key}/runs/{run_id}/image-jobs",
        json={
            "id": str(uuid4()),
            "prompt": f"A portrait of {name}",
            "request_id": str(uuid4()),
            "worker_id": worker_id,
        },
    )
    assert image_job.status_code == 201

    watch = client.post(
        "/api/t3-watches",
        json={
            "id": str(uuid4()),
            "server_id": str(uuid4()),
            "thread_id": f"thread-{key}",
            "origin_agent_key": key,
            "origin_conversation_id": conversation_id,
            "origin_run_id": run_id,
        },
    )
    assert watch.status_code == 201

    return {
        "asset_ids": [atlas_id, note_asset_id],
        "conversation_id": conversation_id,
        "run_id": run_id,
        "watch_id": watch.json()["id"],
    }


def test_delete_agent_permanently_removes_all_agent_data_and_preserves_globals_and_other_agents(
    client,
):
    test_client, data_dir = client
    worker_id = str(uuid4())
    registered = test_client.put(
        "/api/runners/runner-1",
        json={"worker_id": worker_id, "capacity": 5},
    )
    assert registered.status_code == 200

    # Global MCP server and OAuth state must survive agent deletion.
    mcp_server = test_client.post(
        "/api/mcp-servers",
        json={
            "slug": "t3-code",
            "name": "T3 Code",
            "preset": "t3_code",
            "url": "http://127.0.0.1:3773/mcp",
            "auth_type": "oauth",
        },
    )
    assert mcp_server.status_code == 201
    server_id = mcp_server.json()["id"]

    oauth_state = test_client.post(
        "/api/mcp-oauth-states",
        json={
            "state": "state-" + "a" * 24,
            "server_id": server_id,
            "code_verifier": "verifier-" + "b" * 40,
            "redirect_uri": "http://localhost:8630/api/mcp/oauth/callback",
            "token_endpoint": "http://localhost:3773/oauth/token",
            "client_id": "kadan-client",
            "expires_at": (datetime.now(UTC) + timedelta(minutes=10)).isoformat(),
        },
    )
    assert oauth_state.status_code == 201

    kadan = _seed_agent(test_client, "kadan", "Kadan Agent", worker_id)
    other = _seed_agent(test_client, "other", "Other Agent", worker_id)

    # Simulate one already-missing blob file on disk so deletion ignores missing files safely.
    missing_blob = data_dir / "files" / kadan["asset_ids"][1]
    assert missing_blob.is_file()
    missing_blob.unlink()
    present_blob = data_dir / "files" / kadan["asset_ids"][0]
    assert present_blob.is_file()

    deleted = test_client.delete("/api/agents/kadan")
    assert deleted.status_code == 204
    assert deleted.content == b""

    # Deleted agent is completely gone.
    assert test_client.get("/api/agents/kadan").status_code == 404
    assert [a["key"] for a in test_client.get("/api/agents").json()] == ["other"]

    # Stored asset blobs for the deleted agent are removed from disk.
    for asset_id in kadan["asset_ids"]:
        assert not (data_dir / "files" / asset_id).exists()

    # Verify every SQLite table has zero rows for "kadan".
    with sqlite3.connect(data_dir / "tsuchi.sqlite3") as db:
        for table, column in [
            ("agents", "key"),
            ("runner_assignments", "agent_key"),
            ("resources", "agent_key"),
            ("resource_search", "agent_key"),
            ("revisions", "agent_key"),
            ("assets", "agent_key"),
            ("pet_jobs", "agent_key"),
            ("agent_runs", "agent_key"),
            ("image_jobs", "agent_key"),
            ("activity", "agent_key"),
        ]:
            count = db.execute(
                f"SELECT count(*) FROM {table} WHERE {column} = ?", ("kadan",)
            ).fetchone()[0]
            assert count == 0, f"Expected 0 rows for kadan in {table}, got {count}"

        watch_count = db.execute(
            "SELECT count(*) FROM t3_thread_watches WHERE json_extract(data, '$.origin_agent_key') = ?",
            ("kadan",),
        ).fetchone()[0]
        assert watch_count == 0

    # Other agent's records, files, runner assignment, and T3 watch remain intact.
    assert test_client.get("/api/agents/other").status_code == 200
    assert test_client.get("/api/agents/other/persona").json() is not None
    assert test_client.get("/api/agents/other/pet").json() is not None
    assert len(test_client.get("/api/agents/other/resources").json()) == 6
    for asset_id in other["asset_ids"]:
        assert (data_dir / "files" / asset_id).is_file()
    runners = test_client.get("/api/runners").json()
    assert runners[0]["agent_keys"] == ["other"]
    watches = test_client.get("/api/t3-watches").json()
    assert [w["id"] for w in watches] == [other["watch_id"]]

    # Global MCP servers and OAuth states remain intact.
    servers = test_client.get("/api/mcp-servers").json()
    assert [s["id"] for s in servers] == [server_id]
    consumed = test_client.post(
        f"/api/mcp-oauth-states/{oauth_state.json()['state']}/consume"
    )
    assert consumed.status_code == 200

    # Recreating the same agent key ("kadan") immediately after deletion works cleanly.
    recreated = test_client.post(
        "/api/agents", json={"key": "kadan", "name": "Kadan Agent"}
    )
    assert recreated.status_code == 201
    assert recreated.json()["persona_id"] is None
    assert recreated.json()["selected_pet_id"] is None
    assert test_client.get("/api/agents/kadan/resources").json() == []
    assert test_client.get("/api/agents/kadan/assets").json() == []


def test_delete_missing_agent_returns_404(client):
    test_client, _ = client
    response = test_client.delete("/api/agents/does-not-exist")
    assert response.status_code == 404


def test_delete_agent_clears_in_memory_run_draft_on_recreation(client):
    test_client, _ = client
    worker_id = str(uuid4())
    assert (
        test_client.put(
            "/api/runners/runner-1",
            json={"worker_id": worker_id, "capacity": 5},
        ).status_code
        == 200
    )
    assert (
        test_client.post(
            "/api/agents", json={"key": "kadan", "name": "Kadan Agent"}
        ).status_code
        == 201
    )
    assert (
        test_client.post(
            "/api/runners/runner-1/assign", json={"worker_id": worker_id}
        ).status_code
        == 200
    )
    conv = test_client.post(
        "/api/agents/kadan/resources",
        json={"kind": "conversation", "title": "New chat"},
    ).json()
    msg_id = str(uuid5(uuid4(), "reused-run"))
    run = test_client.post(
        f"/api/agents/kadan/resources/{conv['id']}/runs",
        json={
            "expected_version": 1,
            "message": {
                "id": msg_id,
                "role": "user",
                "content": "Hi",
                "created_at": datetime.now(UTC).isoformat(),
            },
        },
    ).json()
    assert (
        test_client.put(
            f"/api/agents/kadan/runs/{run['id']}",
            json={
                "transcript_format": 2,
                "expected_version": 1,
                "worker_id": worker_id,
                "status": "running",
                "owner_id": worker_id,
                "lease_expires_at": (
                    datetime.now(UTC) + timedelta(minutes=2)
                ).isoformat(),
            },
        ).status_code
        == 200
    )
    assert (
        test_client.put(
            f"/api/agents/kadan/runs/{run['id']}/draft",
            json={"worker_id": worker_id, "content": "Stale draft"},
        ).status_code
        == 204
    )
    assert test_client.get(f"/api/agents/kadan/runs/{run['id']}").json()["draft"] == (
        "Stale draft"
    )

    assert test_client.delete("/api/agents/kadan").status_code == 204

    # Recreate agent and reuse the exact same run ID — no stale draft may appear.
    assert (
        test_client.post(
            "/api/agents", json={"key": "kadan", "name": "Kadan Agent"}
        ).status_code
        == 201
    )
    assert (
        test_client.post(
            "/api/runners/runner-1/assign", json={"worker_id": worker_id}
        ).status_code
        == 200
    )
    conv2 = test_client.post(
        "/api/agents/kadan/resources",
        json={"kind": "conversation", "title": "New chat"},
    ).json()
    run2 = test_client.post(
        f"/api/agents/kadan/resources/{conv2['id']}/runs",
        json={
            "expected_version": 1,
            "message": {
                "id": msg_id,
                "role": "user",
                "content": "Fresh",
                "created_at": datetime.now(UTC).isoformat(),
            },
        },
    ).json()
    assert (
        test_client.put(
            f"/api/agents/kadan/runs/{run2['id']}",
            json={
                "transcript_format": 2,
                "expected_version": 1,
                "worker_id": worker_id,
                "status": "running",
                "owner_id": worker_id,
                "lease_expires_at": (
                    datetime.now(UTC) + timedelta(minutes=2)
                ).isoformat(),
            },
        ).status_code
        == 200
    )
    assert test_client.get(f"/api/agents/kadan/runs/{run2['id']}").json()["draft"] is None


def test_file_storage_delete_agent_leaves_disk_untouched_on_repository_error(tmp_path):
    repository = Mock(data_dir=tmp_path)
    repository.delete_agent.side_effect = StorageError("Agent namespace not found", 404)
    files = FileStorage(repository, max_upload_bytes=1024)
    files.initialize()
    blob_id = uuid4()
    blob = files.blob_path(blob_id)
    blob.write_bytes(b"keep-me")

    with pytest.raises(StorageError) as raised:
        files.delete_agent("missing")
    assert raised.value.status_code == 404
    assert blob.read_bytes() == b"keep-me"
