"""Unit tests for async T3 Code background thread & pending-question wakeups."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from uuid import uuid4, uuid5

import httpx

from kadan.api.config import Settings
from kadan.api.services import LocalServices
from kadan.runner.t3_events import (
    MAX_SUMMARY_CHARS,
    T3ThreadEvents,
    _extract_latest_reply,
    _extract_run_status,
)


def test_t3_watcher_wakes_on_pending_question_then_completion():
    now = datetime.now(UTC).isoformat()
    server_id = uuid4()
    watch_id = uuid4()
    convo_id = uuid4()
    origin_run_id = uuid4()

    watch = {
        "id": str(watch_id),
        "server_id": str(server_id),
        "thread_id": "th-99",
        "run_id": "run-1",
        "environment_id": None,
        "thread_link": "https://t3.example/threads/th-99",
        "thread_title": "Ship MCP",
        "origin_agent_key": "mi",
        "origin_conversation_id": str(convo_id),
        "origin_run_id": str(origin_run_id),
        "status": "watching",
        "last_pending_request_id": None,
        "version": 1,
        "created_at": now,
        "updated_at": now,
    }
    server = {
        "id": str(server_id),
        "slug": "t3-code",
        "name": "T3 Code",
        "preset": "t3_code",
        "url": "https://t3.example/mcp",
        "app_url": "https://t3.example",
        "auth_type": "none",
        "auth_state": {},
        "enabled": True,
        "status": "connected",
        "status_error": None,
        "pinned_tools": ["t3_thread_read", "t3_pending_request"],
        "tools": [
            {"name": "t3_thread_read", "description": "Read thread", "input_schema": {}, "read_only": True, "idempotent": True, "accepts_client_request_id": False},
            {"name": "t3_pending_request", "description": "Pending requests", "input_schema": {}, "read_only": False, "idempotent": False, "accepts_client_request_id": False},
        ],
        "tools_synced_at": now,
        "version": 1,
        "created_at": now,
        "updated_at": now,
    }

    phase = {"step": "question"}
    queued_events: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.url.host == "tsuchi.test":
            if path == "/api/t3-watches" and request.method == "GET":
                return httpx.Response(200, json=[watch] if watch["status"] == "watching" else [])
            if path == "/api/mcp-servers" and request.method == "GET":
                return httpx.Response(200, json=[server])
            if path == f"/api/t3-watches/{watch_id}" and request.method == "PUT":
                body = json.loads(request.content)
                assert body["expected_version"] == watch["version"]
                watch.update(
                    {
                        "status": body["status"],
                        "last_pending_request_id": body["last_pending_request_id"],
                        "version": watch["version"] + 1,
                    }
                )
                return httpx.Response(200, json=watch)
            if path == f"/api/agents/mi/resources/{convo_id}/event-runs" and request.method == "POST":
                body = json.loads(request.content)
                queued_events.append(body)
                return httpx.Response(201, json={"id": body["id"]})
            return httpx.Response(404)

        if path == "/mcp":
            rpc = json.loads(request.content)
            method = rpc.get("method")
            if method == "initialize":
                return httpx.Response(200, json={"jsonrpc": "2.0", "id": rpc["id"], "result": {}})
            if method == "notifications/initialized":
                return httpx.Response(202)
            if method == "tools/call":
                name = rpc["params"]["name"]
                args = rpc["params"]["arguments"]
                if name == "t3_pending_request" and args["action"] == "list":
                    data = {"requestIds": ["req-abc"]} if phase["step"] == "question" else {"requestIds": []}
                elif name == "t3_pending_request" and args["action"] == "read":
                    data = {"requestId": "req-abc", "question": "Keep old endpoint?"}
                elif name == "t3_thread_read":
                    data = {
                        "status": "completed" if phase["step"] == "done" else "running",
                        "link": "https://t3.example/threads/th-99",
                        "title": "Ship MCP",
                        "messages": [{"role": "assistant", "text": "All unit tests pass!"}],
                    }
                else:
                    data = {}
                return httpx.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": rpc["id"],
                        "result": {"content": [{"type": "text", "text": json.dumps(data)}]},
                    },
                )
        return httpx.Response(404)

    async def scenario():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        services = LocalServices(
            client, Settings(tsuchi_url="http://tsuchi.test", hikari_url="http://hikari.test")
        )
        watcher = T3ThreadEvents(services)

        # 1. Pending question wakes conversation without terminating watch
        await watcher.poll_once()
        assert len(queued_events) == 1
        assert queued_events[0]["id"] == str(uuid5(watch_id, "t3-q-req-abc"))
        assert queued_events[0]["event"]["kind"] == "t3"
        q_payload = json.loads(queued_events[0]["event"]["content"])
        assert q_payload["event_type"] == "pending_question"
        assert q_payload["question"]["question"] == "Keep old endpoint?"
        assert watch["status"] == "watching"

        # 2. Second poll while same question is still open does NOT duplicate wakeup
        await watcher.poll_once()
        assert len(queued_events) == 1

        # 3. Once thread completes, wakes conversation and marks watch delivered
        phase["step"] = "done"
        await watcher.poll_once()
        assert len(queued_events) == 2
        assert queued_events[1]["id"] == str(uuid5(watch_id, "t3-done"))
        done_payload = json.loads(queued_events[1]["event"]["content"])
        assert done_payload["event_type"] == "thread_settled"
        assert done_payload["status"] == "completed"
        assert done_payload["message"] == "All unit tests pass!"
        assert watch["status"] == "delivered"
        await client.aclose()

    asyncio.run(scenario())


def test_t3_watcher_drops_watch_when_server_deleted_or_conversation_trashed():
    now = datetime.now(UTC).isoformat()
    server_id = uuid4()
    missing_server_id = uuid4()
    watch_missing_server_id = uuid4()
    watch_trashed_convo_id = uuid4()
    convo_id = uuid4()
    origin_run_id = uuid4()

    watches = {
        str(watch_missing_server_id): {
            "id": str(watch_missing_server_id),
            "server_id": str(missing_server_id),
            "thread_id": "th-orphan",
            "run_id": None,
            "environment_id": None,
            "thread_link": None,
            "thread_title": "Orphan",
            "origin_agent_key": "mi",
            "origin_conversation_id": str(convo_id),
            "origin_run_id": str(origin_run_id),
            "status": "watching",
            "last_pending_request_id": None,
            "version": 1,
            "created_at": now,
            "updated_at": now,
        },
        str(watch_trashed_convo_id): {
            "id": str(watch_trashed_convo_id),
            "server_id": str(server_id),
            "thread_id": "th-trashed",
            "run_id": None,
            "environment_id": None,
            "thread_link": None,
            "thread_title": "Trashed",
            "origin_agent_key": "mi",
            "origin_conversation_id": str(convo_id),
            "origin_run_id": str(origin_run_id),
            "status": "watching",
            "last_pending_request_id": None,
            "version": 1,
            "created_at": now,
            "updated_at": now,
        },
    }
    server = {
        "id": str(server_id),
        "slug": "t3-code",
        "name": "T3 Code",
        "preset": "t3_code",
        "url": "https://t3.example/mcp",
        "app_url": "https://t3.example",
        "auth_type": "none",
        "auth_state": {},
        "enabled": True,
        "status": "connected",
        "status_error": None,
        "pinned_tools": ["t3_thread_read"],
        "tools": [
            {"name": "t3_thread_read", "description": "Read thread", "input_schema": {}, "read_only": True, "idempotent": True, "accepts_client_request_id": False},
        ],
        "tools_synced_at": now,
        "version": 1,
        "created_at": now,
        "updated_at": now,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.url.host == "tsuchi.test":
            if path == "/api/t3-watches" and request.method == "GET":
                return httpx.Response(200, json=[w for w in watches.values() if w["status"] == "watching"])
            if path == "/api/mcp-servers" and request.method == "GET":
                return httpx.Response(200, json=[server])
            if path.startswith("/api/t3-watches/") and request.method == "PUT":
                wid = path.rsplit("/", 1)[-1]
                body = json.loads(request.content)
                watches[wid]["status"] = body["status"]
                watches[wid]["version"] += 1
                return httpx.Response(200, json=watches[wid])
            if path.endswith("/event-runs") and request.method == "POST":
                return httpx.Response(409, json={"detail": "Restore this item from trash before editing it"})
            return httpx.Response(404)
        if path == "/mcp":
            rpc = json.loads(request.content)
            if rpc.get("method") == "initialize":
                return httpx.Response(200, json={"jsonrpc": "2.0", "id": rpc["id"], "result": {}})
            if rpc.get("method") == "notifications/initialized":
                return httpx.Response(202)
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": rpc["id"],
                    "result": {"content": [{"type": "text", "text": json.dumps({"status": "completed", "messages": []})}]},
                },
            )
        return httpx.Response(404)

    async def scenario():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        services = LocalServices(
            client, Settings(tsuchi_url="http://tsuchi.test", hikari_url="http://hikari.test")
        )
        watcher = T3ThreadEvents(services)
        await watcher.poll_once()
        assert watches[str(watch_missing_server_id)]["status"] == "dropped"
        assert watches[str(watch_trashed_convo_id)]["status"] == "dropped"
        await client.aclose()

    asyncio.run(scenario())


def test_t3_watcher_parses_nested_thread_recent_runs_and_typed_items():
    now = datetime.now(UTC).isoformat()
    server_id = uuid4()
    watch_id = uuid4()
    convo_id = uuid4()
    origin_run_id = uuid4()

    watch = {
        "id": str(watch_id),
        "server_id": str(server_id),
        "thread_id": "mcp:launch:abc",
        "run_id": "run:thread:abc:ordinal:1",
        "environment_id": None,
        "thread_link": None,
        "thread_title": "Initial title",
        "origin_agent_key": "mi",
        "origin_conversation_id": str(convo_id),
        "origin_run_id": str(origin_run_id),
        "status": "watching",
        "last_pending_request_id": None,
        "version": 1,
        "created_at": now,
        "updated_at": now,
    }
    server = {
        "id": str(server_id),
        "slug": "t3-code",
        "name": "T3 Code",
        "preset": "t3_code",
        "url": "https://t3.example/mcp",
        "app_url": "https://t3.example",
        "auth_type": "none",
        "auth_state": {},
        "enabled": True,
        "status": "connected",
        "status_error": None,
        "pinned_tools": ["t3_thread_read"],
        "tools": [
            {
                "name": "t3_thread_read",
                "description": "Read thread",
                "input_schema": {},
                "read_only": True,
                "idempotent": True,
                "accepts_client_request_id": False,
            },
        ],
        "tools_synced_at": now,
        "version": 1,
        "created_at": now,
        "updated_at": now,
    }

    read_args: list[dict] = []
    queued_events: list[dict] = []
    watch_updates: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.url.host == "tsuchi.test":
            if path == "/api/t3-watches" and request.method == "GET":
                return httpx.Response(200, json=[watch] if watch["status"] == "watching" else [])
            if path == "/api/mcp-servers" and request.method == "GET":
                return httpx.Response(200, json=[server])
            if path == f"/api/t3-watches/{watch_id}" and request.method == "PUT":
                body = json.loads(request.content)
                watch_updates.append(body)
                watch.update(
                    {
                        "status": body["status"],
                        "thread_link": body["thread_link"],
                        "thread_title": body["thread_title"],
                        "version": watch["version"] + 1,
                    }
                )
                return httpx.Response(200, json=watch)
            if path == f"/api/agents/mi/resources/{convo_id}/event-runs" and request.method == "POST":
                body = json.loads(request.content)
                queued_events.append(body)
                return httpx.Response(201, json={"id": body["id"]})
            return httpx.Response(404)

        if path == "/mcp":
            rpc = json.loads(request.content)
            method = rpc.get("method")
            if method == "initialize":
                return httpx.Response(200, json={"jsonrpc": "2.0", "id": rpc["id"], "result": {}})
            if method == "notifications/initialized":
                return httpx.Response(202)
            if method == "tools/call":
                read_args.append(rpc["params"]["arguments"])
                data = {
                    "thread": {
                        "threadId": "mcp:launch:abc",
                        "title": "Refined Pet Animations Title",
                        "link": "https://app.t3.codes/threads/env/mcp%3Alaunch%3Aabc",
                        "status": "completed",
                    },
                    "recentRuns": [
                        {
                            "runId": "run:thread:abc:ordinal:1",
                            "ordinal": 1,
                            "status": "completed",
                        }
                    ],
                    "items": [
                        {
                            "position": 0,
                            "runId": "run:thread:abc:ordinal:1",
                            "type": "user_message",
                            "text": "Investigate pet animations.",
                        },
                        {
                            "position": 2,
                            "runId": "run:thread:abc:ordinal:1",
                            "type": "assistant_message",
                            "text": "I'll trace where Kadan renders the pet.",
                        },
                        {
                            "position": 25,
                            "runId": "run:thread:abc:ordinal:1",
                            "type": "assistant_message",
                            "text": "Full final report with all nine animations.",
                        },
                    ],
                }
                return httpx.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": rpc["id"],
                        "result": {"content": [{"type": "text", "text": json.dumps(data)}]},
                    },
                )
        return httpx.Response(404)

    async def scenario():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        services = LocalServices(
            client, Settings(tsuchi_url="http://tsuchi.test", hikari_url="http://hikari.test")
        )
        watcher = T3ThreadEvents(services)
        await watcher.poll_once()

        assert read_args == [
            {
                "threadId": "mcp:launch:abc",
                "view": "messages",
                "limit": 5,
                "maxCharsPerItem": MAX_SUMMARY_CHARS,
            }
        ]
        assert len(queued_events) == 1
        payload = json.loads(queued_events[0]["event"]["content"])
        assert payload["status"] == "completed"
        assert payload["thread_link"] == "https://app.t3.codes/threads/env/mcp%3Alaunch%3Aabc"
        assert payload["thread_title"] == "Refined Pet Animations Title"
        assert payload["message"] == "Full final report with all nine animations."
        assert watch_updates[0]["thread_link"] == "https://app.t3.codes/threads/env/mcp%3Alaunch%3Aabc"
        assert watch_updates[0]["thread_title"] == "Refined Pet Animations Title"
        await client.aclose()

    asyncio.run(scenario())


def test_t3_extractors_handle_multi_run_recent_runs_and_avoid_stale_replies():
    payload = {
        "thread": {"status": "failed"},
        "recentRuns": [
            {"runId": "run-1", "ordinal": 1, "status": "completed"},
            {"runId": "run-2", "ordinal": 2, "status": "running"},
        ],
        "items": [
            {"position": 1, "runId": "run-1", "type": "assistant_message", "text": "Old reply"},
        ],
    }
    # Matches target run in recentRuns
    assert _extract_run_status(payload, "run-2") == "running"
    assert _extract_run_status(payload, "run-1") == "completed"
    # Without target_run_id, selects highest ordinal run rather than first entry
    assert _extract_run_status(payload, None) == "running"
    # Does not leak run-1's reply when watching run-2
    assert _extract_latest_reply(payload, "run-2") == ""
    assert _extract_latest_reply(payload, "run-1") == "Old reply"
