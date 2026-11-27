"""Unit tests for Kadan Group Chat directed wake-ups, 1-on-1 isolation, and signed outbound webhooks."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4, uuid5

import httpx

from kadan.api.config import Settings
from kadan.api.pydantic_models.agent_runs import (
    AgentRun,
    RunCall,
    RunEvent,
    RunMessage,
    ToolCommand,
)
from kadan.api.services import LocalServices
from kadan.runner.agent_runs import AgentRuns, RunState, is_group_chat_run
from kadan.runner.group_chat_events import (
    MAX_DELIVERY_ATTEMPTS,
    GroupChatWebhookEvents,
)


def _signed_secret() -> tuple[bytes, str]:
    raw = b"super-secret-key-for-group-chat!"
    return raw, "whsec_" + base64.b64encode(raw).decode("ascii")


def test_outbound_webhook_hmac_signing_headers_retry_backoff_and_suppression():
    raw_secret, whsec = _signed_secret()
    room_id = UUID("00000000-0000-0000-0000-000000000001")
    dot_id = uuid4()
    sub_id = uuid4()
    delivery_id = uuid4()
    now = datetime(2026, 11, 27, 16, 0, 0, tzinfo=UTC)

    state: dict[str, Any] = {
        "room": {
            "id": str(room_id),
            "title": "Group Chat",
            "consecutive_agent_turns": 1,
            "max_consecutive_agent_turns": 4,
            "version": 1,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        },
        "participant": {
            "id": str(dot_id),
            "room_id": str(room_id),
            "handle": "dot",
            "display_name": "Dot",
            "kind": "external_agent",
            "agent_key": None,
            "description": "",
            "enabled": True,
            "revoked_at": None,
            "last_seen_at": None,
            "cursor_seq": 0,
            "version": 1,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        },
        "subscription": {
            "id": str(sub_id),
            "room_id": str(room_id),
            "participant_id": str(dot_id),
            "event_name": "group_chat.turn_requested",
            "filters": {},
            "callback_url": "https://dot.example/webhook",
            "signing_secret": whsec,
            "status": "active",
            "expires_at": (now + timedelta(hours=1)).isoformat(),
            "last_delivered_seq": 0,
            "last_error": None,
            "version": 1,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        },
        "delivery": {
            "id": str(delivery_id),
            "subscription_id": str(sub_id),
            "room_id": str(room_id),
            "participant_id": str(dot_id),
            "seq": 7,
            "message_id": str(uuid4()),
            "payload": {
                "room_id": str(room_id),
                "seq": 7,
                "message": {"content": "Hey @dot check this"},
            },
            "attempts": 0,
            "next_attempt_at": now.isoformat(),
            "status": "pending",
            "last_error": None,
            "version": 1,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        },
        "webhook_status": 503,
        "sent_requests": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "tsuchi.test":
            path = request.url.path
            if path == "/api/group-chat/webhook-deliveries" and request.method == "GET":
                d = state["delivery"]
                return httpx.Response(200, json=[d] if d["status"] == "pending" else [])
            if (
                path == f"/api/group-chat/subscriptions/{sub_id}"
                and request.method == "GET"
            ):
                return httpx.Response(200, json=state["subscription"])
            if (
                path == f"/api/group-chat/subscriptions/{sub_id}"
                and request.method == "PUT"
            ):
                body = json.loads(request.content)
                if body.get("status") is not None:
                    state["subscription"]["status"] = body["status"]
                if "last_error" in body:
                    state["subscription"]["last_error"] = body["last_error"]
                state["subscription"]["version"] += 1
                return httpx.Response(200, json=state["subscription"])
            if path in {
                "/api/group-chat/rooms/default",
                f"/api/group-chat/rooms/{room_id}",
            }:
                return httpx.Response(200, json=state["room"])
            if path == f"/api/group-chat/rooms/{room_id}/participants":
                return httpx.Response(200, json=[state["participant"]])
            if (
                path == f"/api/group-chat/webhook-deliveries/{delivery_id}"
                and request.method == "PUT"
            ):
                body = json.loads(request.content)
                state["delivery"]["status"] = body["status"]
                state["delivery"]["attempts"] = body["attempts"]
                if body.get("next_attempt_at"):
                    state["delivery"]["next_attempt_at"] = body["next_attempt_at"]
                state["delivery"]["last_error"] = body.get("last_error")
                state["delivery"]["version"] += 1
                if body["status"] == "delivered":
                    state["subscription"]["last_delivered_seq"] = state["delivery"][
                        "seq"
                    ]
                    state["subscription"]["last_error"] = None
                    state["subscription"]["version"] += 1
                return httpx.Response(200, json=state["delivery"])
            return httpx.Response(404)

        state["sent_requests"].append(
            {"headers": dict(request.headers), "body": request.content}
        )
        return httpx.Response(state["webhook_status"])

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            services = LocalServices(
                client,
                Settings(
                    tsuchi_url="http://tsuchi.test", hikari_url="http://hikari.test"
                ),
            )
            dispatcher = GroupChatWebhookEvents(services)

            # Attempt 1: transient 503 -> stays pending with backoff + attempts=1.
            await dispatcher.dispatch_pending_webhooks(now=now)
            assert len(state["sent_requests"]) == 1
            sent = state["sent_requests"][0]
            wh_id = sent["headers"]["webhook-id"]
            wh_ts = sent["headers"]["webhook-timestamp"]
            wh_sig = sent["headers"]["webhook-signature"]
            assert wh_id == str(delivery_id)
            assert wh_ts == str(int(now.timestamp()))
            assert sent["headers"]["x-mcp-subscription-id"] == str(sub_id)
            expected_sig = "v1," + base64.b64encode(
                hmac.new(
                    raw_secret,
                    f"{wh_id}.{wh_ts}.".encode("utf-8") + sent["body"],
                    hashlib.sha256,
                ).digest()
            ).decode("ascii")
            assert wh_sig == expected_sig
            parsed_body = json.loads(sent["body"])
            assert parsed_body["eventId"] == str(delivery_id)
            assert parsed_body["name"] == "group_chat.turn_requested"
            assert parsed_body["cursor"] == "7"
            assert state["delivery"]["status"] == "pending"
            assert state["delivery"]["attempts"] == 1
            assert datetime.fromisoformat(state["delivery"]["next_attempt_at"]) > now

            # Attempt 2: succeeds with 200 -> marked delivered and clears subscription error.
            state["webhook_status"] = 200
            await dispatcher.dispatch_pending_webhooks(now=now + timedelta(seconds=10))
            assert state["delivery"]["status"] == "delivered"
            assert state["delivery"]["attempts"] == 2
            assert state["subscription"]["last_delivered_seq"] == 7

            # Turn-cap suppression: pending delivery is dropped without HTTP POST when room is at cap.
            state["delivery"]["status"] = "pending"
            state["delivery"]["attempts"] = 0
            state["room"]["consecutive_agent_turns"] = 4
            sent_before = len(state["sent_requests"])
            await dispatcher.dispatch_pending_webhooks(now=now + timedelta(seconds=20))
            assert len(state["sent_requests"]) == sent_before
            assert state["delivery"]["status"] == "dropped"

            # Revoked participant suppression: drops delivery and revokes subscription without HTTP POST.
            state["room"]["consecutive_agent_turns"] = 1
            state["delivery"]["status"] = "pending"
            state["participant"]["revoked_at"] = now.isoformat()
            state["participant"]["enabled"] = False
            await dispatcher.dispatch_pending_webhooks(now=now + timedelta(seconds=30))
            assert len(state["sent_requests"]) == sent_before
            assert state["delivery"]["status"] == "dropped"
            assert state["subscription"]["status"] == "revoked"

            # Permanent 410 marks both delivery and subscription failed without retries.
            state["participant"]["revoked_at"] = None
            state["participant"]["enabled"] = True
            state["subscription"]["status"] = "active"
            state["delivery"]["status"] = "pending"
            state["delivery"]["attempts"] = MAX_DELIVERY_ATTEMPTS - 1
            state["webhook_status"] = 410
            await dispatcher.dispatch_pending_webhooks(now=now + timedelta(seconds=40))
            assert state["delivery"]["status"] == "failed"
            assert state["subscription"]["status"] == "failed"

    asyncio.run(scenario())


def test_directed_kadan_wakeups_ignore_untargeted_and_coalesce_bursts():
    room_id = UUID("00000000-0000-0000-0000-000000000001")
    kadan_id = uuid4()
    dot_id = uuid4()
    conv_id = uuid4()
    now = datetime.now(UTC).isoformat()

    messages: list[dict[str, Any]] = [
        # Untagged user message -> should NOT wake Kadan.
        {
            "id": str(uuid4()),
            "room_id": str(room_id),
            "seq": 1,
            "sender_kind": "user",
            "participant_id": None,
            "sender_name": "You",
            "content": "FYI everyone",
            "handoff_to_participant_ids": [],
            "handoff_suppressed": False,
            "reply_to_message_id": None,
            "client_request_id": None,
            "created_at": now,
        },
        # Message handed off to Dot only -> should NOT wake Kadan.
        {
            "id": str(uuid4()),
            "room_id": str(room_id),
            "seq": 2,
            "sender_kind": "user",
            "participant_id": None,
            "sender_name": "You",
            "content": "Hey @dot",
            "handoff_to_participant_ids": [str(dot_id)],
            "handoff_suppressed": False,
            "reply_to_message_id": None,
            "client_request_id": None,
            "created_at": now,
        },
    ]
    event_runs: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in {
            "/api/group-chat/rooms/default",
            f"/api/group-chat/rooms/{room_id}",
        }:
            return httpx.Response(
                200,
                json={
                    "id": str(room_id),
                    "title": "Group Chat",
                    "consecutive_agent_turns": 0,
                    "max_consecutive_agent_turns": 6,
                    "version": 1,
                    "created_at": now,
                    "updated_at": now,
                },
            )
        if path == f"/api/group-chat/rooms/{room_id}/participants":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": str(kadan_id),
                        "room_id": str(room_id),
                        "handle": "kadan",
                        "display_name": "Kadan",
                        "kind": "kadan_agent",
                        "agent_key": "kadan",
                        "description": "",
                        "enabled": True,
                        "revoked_at": None,
                        "last_seen_at": None,
                        "cursor_seq": 0,
                        "version": 1,
                        "created_at": now,
                        "updated_at": now,
                    },
                    {
                        "id": str(dot_id),
                        "room_id": str(room_id),
                        "handle": "dot",
                        "display_name": "Dot",
                        "kind": "external_agent",
                        "agent_key": None,
                        "description": "",
                        "enabled": True,
                        "revoked_at": None,
                        "last_seen_at": None,
                        "cursor_seq": 0,
                        "version": 1,
                        "created_at": now,
                        "updated_at": now,
                    },
                ],
            )
        if path == f"/api/group-chat/rooms/{room_id}/messages":
            return httpx.Response(
                200,
                json={
                    "room": {
                        "id": str(room_id),
                        "title": "Group Chat",
                        "consecutive_agent_turns": 0,
                        "max_consecutive_agent_turns": 6,
                        "version": 1,
                        "created_at": now,
                        "updated_at": now,
                    },
                    "messages": messages,
                    "latest_seq": messages[-1]["seq"],
                    "has_more": False,
                },
            )
        if path == "/api/agents/kadan/resources":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": str(conv_id),
                        "agent_key": "kadan",
                        "kind": "conversation",
                        "title": "Chat",
                        "markdown": "",
                        "messages": [],
                        "version": 1,
                        "created_at": now,
                        "updated_at": now,
                    }
                ],
            )
        if path == f"/api/agents/kadan/resources/{conv_id}/runs":
            return httpx.Response(200, json=event_runs)
        if path == f"/api/agents/kadan/resources/{conv_id}/event-runs":
            body = json.loads(request.content)
            run = {
                "id": body["id"],
                "agent_key": "kadan",
                "conversation_id": str(conv_id),
                "trigger_event": body["event"],
                "status": "queued",
                "transcript": [],
                "references": [],
                "delivered_message_ids": [],
                "version": 1,
                "created_at": now,
                "updated_at": now,
            }
            event_runs.append(run)
            return httpx.Response(201, json=run)
        return httpx.Response(404)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            services = LocalServices(
                client,
                Settings(
                    tsuchi_url="http://tsuchi.test", hikari_url="http://hikari.test"
                ),
            )
            dispatcher = GroupChatWebhookEvents(services)

            # Untagged + Dot-only messages do NOT wake Kadan.
            await dispatcher.reconcile_kadan_wakeups()
            assert event_runs == []

            # Burst of 2 messages targeting Kadan queues ONLY ONE coalesced event-run.
            messages.append(
                {
                    "id": str(uuid4()),
                    "room_id": str(room_id),
                    "seq": 3,
                    "sender_kind": "external_agent",
                    "participant_id": str(dot_id),
                    "sender_name": "Dot",
                    "content": "First ping @kadan",
                    "handoff_to_participant_ids": [str(kadan_id)],
                    "handoff_suppressed": False,
                    "reply_to_message_id": None,
                    "client_request_id": None,
                    "created_at": now,
                }
            )
            messages.append(
                {
                    "id": str(uuid4()),
                    "room_id": str(room_id),
                    "seq": 4,
                    "sender_kind": "user",
                    "participant_id": None,
                    "sender_name": "You",
                    "content": "Second ping @kadan",
                    "handoff_to_participant_ids": [str(kadan_id)],
                    "handoff_suppressed": False,
                    "reply_to_message_id": None,
                    "client_request_id": None,
                    "created_at": now,
                }
            )
            await dispatcher.reconcile_kadan_wakeups()
            assert len(event_runs) == 1
            # Polling again while the run is queued does not enqueue a duplicate run.
            await dispatcher.reconcile_kadan_wakeups()
            assert len(event_runs) == 1

    asyncio.run(scenario())


def test_group_chat_run_isolation_full_unread_context_and_replay_idempotency(tmp_path):
    instruction_file = tmp_path / "AGENT.md"
    instruction_file.write_text("Test instructions")
    room_id = UUID("00000000-0000-0000-0000-000000000001")
    kadan_id = uuid4()
    dot_id = uuid4()
    conv_id = uuid4()
    run_id = uuid4()
    owner_id = uuid4()
    now = datetime.now(UTC)

    one_on_one_deliveries: list[dict[str, Any]] = []
    group_chat_posts: list[dict[str, Any]] = []
    cursor_updates: list[int] = []

    run = AgentRun(
        id=run_id,
        agent_key="kadan",
        conversation_id=conv_id,
        trigger_event=RunEvent(
            kind="group_chat",
            content=json.dumps({"room_id": str(room_id), "trigger_seq": 2}),
        ),
        status="running",
        owner_id=owner_id,
        lease_expires_at=now + timedelta(seconds=60),
        version=1,
        created_at=now,
        updated_at=now,
    )
    saved_run: dict[str, Any] = run.model_dump(mode="json")

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method
        if path == "/api/agents/kadan" and method == "GET":
            return httpx.Response(
                200,
                json={"key": "kadan", "name": "Kadan", "created_at": now.isoformat()},
            )
        if path == "/api/agents/kadan/persona" and method == "GET":
            return httpx.Response(
                200, content=b"null", headers={"content-type": "application/json"}
            )
        if path == f"/api/agents/kadan/resources/{conv_id}" and method == "GET":
            return httpx.Response(
                200,
                json={
                    "id": str(conv_id),
                    "agent_key": "kadan",
                    "kind": "conversation",
                    "title": "Chat",
                    "markdown": "",
                    "messages": [],
                    "version": 1,
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                },
            )
        if path == f"/api/agents/kadan/resources/{conv_id}/runs" and method == "GET":
            return httpx.Response(200, json=[saved_run])
        if path == f"/api/agents/kadan/runs/{run_id}" and method == "GET":
            return httpx.Response(200, json=saved_run)
        if path == f"/api/agents/kadan/runs/{run_id}" and method == "PUT":
            body = json.loads(request.content)
            saved_run.update(
                {
                    "status": body["status"],
                    "transcript": body["transcript"],
                    "owner_id": body["owner_id"],
                    "lease_expires_at": body["lease_expires_at"],
                    "error": body["error"],
                    "version": saved_run["version"] + 1,
                }
            )
            return httpx.Response(200, json=saved_run)
        if path == f"/api/agents/kadan/runs/{run_id}/messages" and method == "POST":
            one_on_one_deliveries.append(json.loads(request.content))
            return httpx.Response(201, json={})
        if path == "/api/mcp-servers" and method == "GET":
            return httpx.Response(200, json=[])
        if path in {
            "/api/group-chat/rooms/default",
            f"/api/group-chat/rooms/{room_id}",
        }:
            return httpx.Response(
                200,
                json={
                    "id": str(room_id),
                    "title": "Group Chat",
                    "consecutive_agent_turns": 0,
                    "max_consecutive_agent_turns": 6,
                    "version": 1,
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                },
            )
        if path == f"/api/group-chat/rooms/{room_id}/participants":
            kadan_p = {
                "id": str(kadan_id),
                "room_id": str(room_id),
                "handle": "kadan",
                "display_name": "Kadan",
                "kind": "kadan_agent",
                "agent_key": "kadan",
                "description": "",
                "enabled": True,
                "revoked_at": None,
                "last_seen_at": None,
                "cursor_seq": 0,
                "version": 1,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
            }
            dot_p = {
                "id": str(dot_id),
                "room_id": str(room_id),
                "handle": "dot",
                "display_name": "Dot",
                "kind": "external_agent",
                "agent_key": None,
                "description": "",
                "enabled": True,
                "revoked_at": None,
                "last_seen_at": None,
                "cursor_seq": 0,
                "version": 1,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
            }
            if method == "POST":
                return httpx.Response(201, json=kadan_p)
            return httpx.Response(200, json=[kadan_p, dot_p])
        if path == f"/api/group-chat/rooms/{room_id}/participants/{kadan_id}/touch":
            body = json.loads(request.content)
            if body.get("cursor_seq") is not None:
                cursor_updates.append(body["cursor_seq"])
            return httpx.Response(
                200,
                json={
                    "id": str(kadan_id),
                    "room_id": str(room_id),
                    "handle": "kadan",
                    "display_name": "Kadan",
                    "kind": "kadan_agent",
                    "agent_key": "kadan",
                    "description": "",
                    "enabled": True,
                    "revoked_at": None,
                    "last_seen_at": now.isoformat(),
                    "cursor_seq": body.get("cursor_seq") or 0,
                    "version": 1,
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                },
            )
        if path == f"/api/group-chat/rooms/{room_id}/messages" and method == "GET":
            return httpx.Response(
                200,
                json={
                    "room": {
                        "id": str(room_id),
                        "title": "Group Chat",
                        "consecutive_agent_turns": 0,
                        "max_consecutive_agent_turns": 6,
                        "version": 1,
                        "created_at": now.isoformat(),
                        "updated_at": now.isoformat(),
                    },
                    "messages": [
                        {
                            "id": str(uuid4()),
                            "room_id": str(room_id),
                            "seq": 1,
                            "sender_kind": "user",
                            "participant_id": None,
                            "sender_name": "You",
                            "content": "Context message before tag",
                            "handoff_to_participant_ids": [],
                            "handoff_suppressed": False,
                            "reply_to_message_id": None,
                            "client_request_id": None,
                            "created_at": now.isoformat(),
                        },
                        {
                            "id": str(uuid4()),
                            "room_id": str(room_id),
                            "seq": 2,
                            "sender_kind": "external_agent",
                            "participant_id": str(dot_id),
                            "sender_name": "Dot",
                            "content": "Tagging @kadan now",
                            "handoff_to_participant_ids": [str(kadan_id)],
                            "handoff_suppressed": False,
                            "reply_to_message_id": None,
                            "client_request_id": None,
                            "created_at": now.isoformat(),
                        },
                    ],
                    "latest_seq": 2,
                    "has_more": False,
                },
            )
        if path == f"/api/group-chat/rooms/{room_id}/messages" and method == "POST":
            body = json.loads(request.content)
            group_chat_posts.append(body)
            return httpx.Response(
                201,
                json={
                    "message": {
                        "id": str(uuid4()),
                        "room_id": str(room_id),
                        "seq": 3,
                        "sender_kind": "kadan_agent",
                        "participant_id": str(kadan_id),
                        "sender_name": "Kadan",
                        "content": body["content"],
                        "handoff_to_participant_ids": body.get(
                            "handoff_to_participant_ids", []
                        ),
                        "handoff_suppressed": False,
                        "reply_to_message_id": None,
                        "client_request_id": body.get("client_request_id"),
                        "created_at": now.isoformat(),
                    },
                    "room": {
                        "id": str(room_id),
                        "title": "Group Chat",
                        "consecutive_agent_turns": 1,
                        "max_consecutive_agent_turns": 6,
                        "version": 2,
                        "created_at": now.isoformat(),
                        "updated_at": now.isoformat(),
                    },
                    "effective_handoff_to_participant_ids": body.get(
                        "handoff_to_participant_ids", []
                    ),
                    "turn_cap_reached": False,
                    "handoff_suppressed": False,
                    "idempotent_replay": False,
                    "enqueued_delivery_ids": [],
                },
            )
        return httpx.Response(404, json={"detail": f"Unhandled {method} {path}"})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            settings = Settings(
                tsuchi_url="http://tsuchi.test",
                hikari_url="http://hikari.test",
                agent_instruction_file=str(instruction_file),
            )
            services = LocalServices(client, settings)
            runs = AgentRuns(services, owner=owner_id)

            assert is_group_chat_run(run) is True
            # 1-on-1 isolation: send_message is NOT bound on group_chat runs; Group Chat tools ARE bound.
            tools = runs.bound_tools(run)
            tool_names = {t.name for t in tools}
            assert "send_message" not in tool_names
            assert "group_chat_send_message" in tool_names
            assert "group_chat_read_messages" in tool_names
            assert "group_chat_list_participants" in tool_names
            assert "finish" in tool_names

            # Context injects group_chat_event with FULL unread history (both untagged seq 1 and tagged seq 2).
            ctx = await runs.context(run)
            obs = [
                item
                for item in ctx
                if getattr(item, "name", None) == "group_chat_event"
            ]
            assert len(obs) == 1
            assert obs[0].data["latest_seq"] == 2
            assert [m["seq"] for m in obs[0].data["unread_messages"]] == [1, 2]
            assert obs[0].data["addressed_by"][0]["sender_name"] == "Dot"

            # Invoking group_chat_send_message passes deterministic client_request_id from action_id and ends run.
            cmd = ToolCommand(
                tool="group_chat_send_message",
                arguments={"content": "Hi Dot!", "handoff_to": "dot"},
            )
            transcript = [
                RunMessage(role="system", content="sys"),
                RunCall(command=cmd),
            ]
            saved_run["transcript"] = [
                item.model_dump(mode="json") for item in transcript
            ]
            run_with_call = AgentRun.model_validate(saved_run)
            expected_action_id = str(uuid5(run_id, f"command:{len(transcript)}"))

            out = await runs.invoke_tool(
                RunState(run=run_with_call, command=cmd, replayed=True)
            )
            assert out["run"].status == "completed"
            assert len(group_chat_posts) == 1
            assert group_chat_posts[0]["client_request_id"] == expected_action_id
            assert group_chat_posts[0]["handoff_to_participant_ids"] == [str(dot_id)]
            # Zero messages leaked into 1-on-1 Chat!
            assert one_on_one_deliveries == []

            # Even on failure, a group_chat run NEVER leaks a fallback reply into 1-on-1 Chat.
            saved_run["status"] = "running"
            saved_run["owner_id"] = str(owner_id)
            saved_run["lease_expires_at"] = (now + timedelta(seconds=60)).isoformat()
            failing_run = AgentRun.model_validate(saved_run)
            await runs.handle_failure(failing_run, RuntimeError("boom"))
            assert saved_run["status"] == "failed"
            assert one_on_one_deliveries == []

    asyncio.run(scenario())


def test_group_chat_cursor_advances_only_through_included_page_and_reconciles_overflow(
    tmp_path,
):
    """A directed @mention beyond the 100-row hydration cap must not be skipped."""
    instruction_file = tmp_path / "AGENT.md"
    instruction_file.write_text("Test instructions")
    room_id = UUID("00000000-0000-0000-0000-000000000001")
    kadan_id = uuid4()
    dot_id = uuid4()
    conv_id = uuid4()
    owner_id = uuid4()
    now = datetime.now(UTC)

    # 105 messages: seq 1..100 untargeted, seq 105 explicitly tags @kadan.
    all_messages: list[dict[str, Any]] = []
    for seq in range(1, 106):
        all_messages.append(
            {
                "id": str(uuid4()),
                "room_id": str(room_id),
                "seq": seq,
                "sender_kind": "user",
                "participant_id": None,
                "sender_name": "You",
                "content": f"Message {seq}"
                if seq < 105
                else "Hey @kadan please answer #105",
                "handoff_to_participant_ids": [str(kadan_id)] if seq == 105 else [],
                "handoff_suppressed": False,
                "reply_to_message_id": None,
                "client_request_id": None,
                "created_at": now.isoformat(),
            }
        )

    state: dict[str, Any] = {
        "cursor_seq": 0,
        "event_runs": [],
    }

    def kadan_participant_json() -> dict[str, Any]:
        return {
            "id": str(kadan_id),
            "room_id": str(room_id),
            "handle": "kadan",
            "display_name": "Kadan",
            "kind": "kadan_agent",
            "agent_key": "kadan",
            "description": "",
            "enabled": True,
            "revoked_at": None,
            "last_seen_at": None,
            "cursor_seq": state["cursor_seq"],
            "version": 1,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method
        if path == "/api/agents/kadan" and method == "GET":
            return httpx.Response(
                200,
                json={"key": "kadan", "name": "Kadan", "created_at": now.isoformat()},
            )
        if path == "/api/agents/kadan/persona" and method == "GET":
            return httpx.Response(
                200, content=b"null", headers={"content-type": "application/json"}
            )
        if path == "/api/agents/kadan/resources" and method == "GET":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": str(conv_id),
                        "agent_key": "kadan",
                        "kind": "conversation",
                        "title": "Chat",
                        "markdown": "",
                        "messages": [],
                        "version": 1,
                        "created_at": now.isoformat(),
                        "updated_at": now.isoformat(),
                    }
                ],
            )
        if path == f"/api/agents/kadan/resources/{conv_id}" and method == "GET":
            return httpx.Response(
                200,
                json={
                    "id": str(conv_id),
                    "agent_key": "kadan",
                    "kind": "conversation",
                    "title": "Chat",
                    "markdown": "",
                    "messages": [],
                    "version": 1,
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                },
            )
        if path == f"/api/agents/kadan/resources/{conv_id}/runs" and method == "GET":
            return httpx.Response(200, json=state["event_runs"])
        if (
            path == f"/api/agents/kadan/resources/{conv_id}/event-runs"
            and method == "POST"
        ):
            body = json.loads(request.content)
            for existing in state["event_runs"]:
                if existing["id"] == body["id"]:
                    return httpx.Response(201, json=existing)
            run_rec = {
                "id": body["id"],
                "agent_key": "kadan",
                "conversation_id": str(conv_id),
                "trigger_event": body["event"],
                "status": "queued",
                "transcript": [],
                "references": [],
                "delivered_message_ids": [],
                "version": 1,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
            }
            state["event_runs"].append(run_rec)
            return httpx.Response(201, json=run_rec)
        if path == "/api/mcp-servers" and method == "GET":
            return httpx.Response(200, json=[])
        if path in {
            "/api/group-chat/rooms/default",
            f"/api/group-chat/rooms/{room_id}",
        }:
            return httpx.Response(
                200,
                json={
                    "id": str(room_id),
                    "title": "Group Chat",
                    "consecutive_agent_turns": 0,
                    "max_consecutive_agent_turns": 6,
                    "version": 1,
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                },
            )
        if path == f"/api/group-chat/rooms/{room_id}/participants":
            if method == "POST":
                return httpx.Response(201, json=kadan_participant_json())
            return httpx.Response(
                200,
                json=[
                    kadan_participant_json(),
                    {
                        "id": str(dot_id),
                        "room_id": str(room_id),
                        "handle": "dot",
                        "display_name": "Dot",
                        "kind": "external_agent",
                        "agent_key": None,
                        "description": "",
                        "enabled": True,
                        "revoked_at": None,
                        "last_seen_at": None,
                        "cursor_seq": 0,
                        "version": 1,
                        "created_at": now.isoformat(),
                        "updated_at": now.isoformat(),
                    },
                ],
            )
        if path == f"/api/group-chat/rooms/{room_id}/participants/{kadan_id}/touch":
            body = json.loads(request.content)
            if body.get("cursor_seq") is not None:
                state["cursor_seq"] = max(state["cursor_seq"], body["cursor_seq"])
            return httpx.Response(200, json=kadan_participant_json())
        if path == f"/api/group-chat/rooms/{room_id}/messages" and method == "GET":
            after_seq = int(request.url.params.get("after_seq", "0"))
            limit = int(request.url.params.get("limit", "50"))
            unread = [m for m in all_messages if m["seq"] > after_seq]
            page = unread[:limit]
            return httpx.Response(
                200,
                json={
                    "room": {
                        "id": str(room_id),
                        "title": "Group Chat",
                        "consecutive_agent_turns": 0,
                        "max_consecutive_agent_turns": 6,
                        "version": 1,
                        "created_at": now.isoformat(),
                        "updated_at": now.isoformat(),
                    },
                    "messages": page,
                    "latest_seq": all_messages[-1]["seq"],
                    "has_more": len(unread) > limit,
                },
            )
        return httpx.Response(404, json={"detail": f"Unhandled {method} {path}"})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            settings = Settings(
                tsuchi_url="http://tsuchi.test",
                hikari_url="http://hikari.test",
                agent_instruction_file=str(instruction_file),
            )
            services = LocalServices(client, settings)
            runs = AgentRuns(services, owner=owner_id)
            dispatcher = GroupChatWebhookEvents(services)

            # 1. Reconciler scans past the first 100-row page and wakes Kadan for seq 105.
            await dispatcher.reconcile_kadan_wakeups()
            assert len(state["event_runs"]) == 1
            first_run_rec = state["event_runs"][0]
            first_run = AgentRun.model_validate(
                {
                    **first_run_rec,
                    "status": "running",
                    "owner_id": str(owner_id),
                    "lease_expires_at": (now + timedelta(seconds=60)).isoformat(),
                }
            )

            # 2. First wake hydrates only seq 1..100; cursor advances ONLY to 100, NOT room latest_seq 105!
            ctx1 = await runs.context(first_run)
            obs1 = next(
                item
                for item in ctx1
                if getattr(item, "name", None) == "group_chat_event"
            )
            assert obs1.data["latest_seq"] == 105
            assert obs1.data["last_included_seq"] == 100
            assert obs1.data["has_more"] is True
            assert len(obs1.data["unread_messages"]) == 100
            await runs._advance_group_chat_cursor(
                first_run.model_copy(update={"transcript": ctx1})
            )
            assert state["cursor_seq"] == 100
            first_run_rec["status"] = "completed"

            # 3. Subsequent reconciliation sees seq 105 still unread after cursor 100 and queues a new wake run.
            await dispatcher.reconcile_kadan_wakeups()
            assert len(state["event_runs"]) == 2
            second_run_rec = state["event_runs"][1]
            assert second_run_rec["id"] != first_run_rec["id"]
            second_run = AgentRun.model_validate(
                {
                    **second_run_rec,
                    "status": "running",
                    "owner_id": str(owner_id),
                    "lease_expires_at": (now + timedelta(seconds=60)).isoformat(),
                }
            )

            # 4. Second wake receives seq 101..105 including the directed @mention at seq 105, then advances to 105.
            ctx2 = await runs.context(second_run)
            obs2 = next(
                item
                for item in ctx2
                if getattr(item, "name", None) == "group_chat_event"
            )
            assert obs2.data["last_included_seq"] == 105
            assert obs2.data["has_more"] is False
            assert [m["seq"] for m in obs2.data["unread_messages"]] == [
                101,
                102,
                103,
                104,
                105,
            ]
            assert obs2.data["addressed_by"][0]["seq"] == 105
            await runs._advance_group_chat_cursor(
                second_run.model_copy(update={"transcript": ctx2})
            )
            assert state["cursor_seq"] == 105
            second_run_rec["status"] = "completed"

            # 5. No further wake runs once caught up to 105.
            await dispatcher.reconcile_kadan_wakeups()
            assert len(state["event_runs"]) == 2

    asyncio.run(scenario())
