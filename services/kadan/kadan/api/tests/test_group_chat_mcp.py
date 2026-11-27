"""Unit tests for the Group Chat Streamable HTTP MCP server, MCP Events, and browser API routes."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import httpx
from fastapi.testclient import TestClient

from kadan.api.config import Settings
from kadan.api.group_chat_client import hash_group_chat_token
from kadan.api.group_chat_mcp import GroupChatMcpHandler, GroupChatRateLimiter
from kadan.api.server import create_app
from kadan.api.services import LocalServices


class FakeTsuchiGroupChat:
    """In-memory Tsuchi Group Chat simulator for Kadan API/MCP unit tests."""

    def __init__(self, *, max_turns: int = 3):
        now = datetime.now(UTC)
        self.room_id = UUID("00000000-0000-0000-0000-000000000001")
        self.room: dict[str, Any] = {
            "id": str(self.room_id),
            "title": "Group Chat",
            "consecutive_agent_turns": 0,
            "max_consecutive_agent_turns": max_turns,
            "version": 1,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }
        self.agents = [{"key": "kadan", "name": "Kadan", "created_at": now.isoformat()}]
        self.conv_id = uuid4()
        self.participants: dict[str, dict[str, Any]] = {}
        self.tokens_by_hash: dict[str, str] = {}
        self.messages: list[dict[str, Any]] = []
        self.subscriptions: dict[str, dict[str, Any]] = {}
        self.event_runs: list[dict[str, Any]] = []
        self.webhook_deliveries: list[dict[str, Any]] = []

    def add_participant(
        self,
        *,
        display_name: str,
        kind: str,
        handle: str | None = None,
        agent_key: str | None = None,
        description: str = "",
        raw_token: str | None = None,
        enabled: bool = True,
    ) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        pid = str(uuid4())
        derived = handle or display_name.lower().replace(" ", "-")
        rec = {
            "id": pid,
            "room_id": str(self.room_id),
            "handle": derived,
            "display_name": display_name,
            "kind": kind,
            "agent_key": agent_key,
            "description": description,
            "enabled": enabled,
            "revoked_at": None,
            "last_seen_at": None,
            "cursor_seq": 0,
            "version": 1,
            "created_at": now,
            "updated_at": now,
        }
        self.participants[pid] = rec
        if raw_token is not None:
            self.tokens_by_hash[hash_group_chat_token(raw_token)] = pid
        return rec

    def handle_tsuchi(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method
        now = datetime.now(UTC).isoformat()

        if path == "/api/agents" and method == "GET":
            return httpx.Response(200, json=self.agents)
        if path == "/api/agents/kadan" and method == "GET":
            return httpx.Response(200, json=self.agents[0])
        if path == "/api/agents/kadan/resources" and method == "GET":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": str(self.conv_id),
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
        if (
            path == f"/api/agents/kadan/resources/{self.conv_id}/runs"
            and method == "GET"
        ):
            return httpx.Response(200, json=self.event_runs)
        if (
            path == f"/api/agents/kadan/resources/{self.conv_id}/event-runs"
            and method == "POST"
        ):
            body = json.loads(request.content)
            run = {
                "id": body["id"],
                "agent_key": "kadan",
                "conversation_id": str(self.conv_id),
                "trigger_event": body["event"],
                "status": "queued",
                "transcript": [],
                "references": [],
                "delivered_message_ids": [],
                "version": 1,
                "created_at": now,
                "updated_at": now,
            }
            self.event_runs.append(run)
            return httpx.Response(201, json=run)

        if path in {
            "/api/group-chat/rooms/default",
            f"/api/group-chat/rooms/{self.room_id}",
        }:
            if method == "GET":
                return httpx.Response(200, json=self.room)
            if method == "PUT":
                body = json.loads(request.content)
                if body.get("title") is not None:
                    self.room["title"] = body["title"]
                if body.get("max_consecutive_agent_turns") is not None:
                    self.room["max_consecutive_agent_turns"] = body[
                        "max_consecutive_agent_turns"
                    ]
                self.room["version"] += 1
                self.room["updated_at"] = now
                return httpx.Response(200, json=self.room)

        if path == f"/api/group-chat/rooms/{self.room_id}/participants":
            if method == "GET":
                return httpx.Response(200, json=list(self.participants.values()))
            if method == "POST":
                body = json.loads(request.content)
                if body["kind"] == "kadan_agent":
                    for existing in self.participants.values():
                        if existing["kind"] == "kadan_agent" and existing[
                            "agent_key"
                        ] == body.get("agent_key"):
                            return httpx.Response(201, json=existing)
                pid = str(uuid4())
                handle = body.get("handle") or body[
                    "display_name"
                ].strip().lower().replace(" ", "-")
                rec = {
                    "id": pid,
                    "room_id": str(self.room_id),
                    "handle": handle,
                    "display_name": body["display_name"],
                    "kind": body["kind"],
                    "agent_key": body.get("agent_key"),
                    "description": body.get("description", ""),
                    "enabled": body.get("enabled", True),
                    "revoked_at": None,
                    "last_seen_at": None,
                    "cursor_seq": 0,
                    "version": 1,
                    "created_at": now,
                    "updated_at": now,
                }
                self.participants[pid] = rec
                if body.get("token_hash"):
                    self.tokens_by_hash[body["token_hash"]] = pid
                return httpx.Response(201, json=rec)

        if path.startswith(f"/api/group-chat/rooms/{self.room_id}/participants/"):
            rest = path.removeprefix(
                f"/api/group-chat/rooms/{self.room_id}/participants/"
            )
            if rest.endswith("/touch") and method == "POST":
                pid = rest.removesuffix("/touch")
                rec = self.participants[pid]
                body = json.loads(request.content)
                if body.get("last_seen_at"):
                    rec["last_seen_at"] = body["last_seen_at"]
                if body.get("cursor_seq") and body["cursor_seq"] > rec["cursor_seq"]:
                    rec["cursor_seq"] = body["cursor_seq"]
                return httpx.Response(200, json=rec)
            pid = rest
            if method == "PUT":
                rec = self.participants[pid]
                body = json.loads(request.content)
                if body.get("display_name") is not None:
                    rec["display_name"] = body["display_name"]
                if body.get("description") is not None:
                    rec["description"] = body["description"]
                if body.get("revoke"):
                    rec["enabled"] = False
                    rec["revoked_at"] = now
                    for token_hash, owner in list(self.tokens_by_hash.items()):
                        if owner == pid:
                            del self.tokens_by_hash[token_hash]
                    for sub in self.subscriptions.values():
                        if sub["participant_id"] == pid and sub["status"] == "active":
                            sub["status"] = "revoked"
                elif body.get("enabled") is not None:
                    rec["enabled"] = body["enabled"]
                if body.get("token_hash"):
                    for token_hash, owner in list(self.tokens_by_hash.items()):
                        if owner == pid:
                            del self.tokens_by_hash[token_hash]
                    self.tokens_by_hash[body["token_hash"]] = pid
                rec["version"] += 1
                rec["updated_at"] = now
                return httpx.Response(200, json=rec)

        if path == "/api/group-chat/participants/authenticate" and method == "POST":
            body = json.loads(request.content)
            pid = self.tokens_by_hash.get(body["token_hash"])
            if not pid:
                return httpx.Response(
                    401,
                    json={
                        "detail": "Invalid group chat bearer token",
                        "code": "invalid_token",
                    },
                )
            rec = self.participants[pid]
            if rec["revoked_at"] is not None:
                return httpx.Response(
                    403,
                    json={
                        "detail": "This agent's access has been revoked",
                        "code": "participant_revoked",
                    },
                )
            if not rec["enabled"]:
                return httpx.Response(
                    403,
                    json={
                        "detail": "This agent is currently paused",
                        "code": "participant_paused",
                    },
                )
            rec["last_seen_at"] = now
            return httpx.Response(200, json=rec)

        if path == f"/api/group-chat/rooms/{self.room_id}/messages":
            if method == "GET":
                after_seq = request.url.params.get("after_seq")
                before_seq = request.url.params.get("before_seq")
                limit = int(request.url.params.get("limit", "50"))
                msgs = list(self.messages)
                if after_seq is not None:
                    msgs = [m for m in msgs if m["seq"] > int(after_seq)]
                if before_seq is not None:
                    msgs = [m for m in msgs if m["seq"] < int(before_seq)]
                has_more = len(msgs) > limit
                msgs = msgs[:limit] if after_seq is not None else msgs[-limit:]
                latest_seq = self.messages[-1]["seq"] if self.messages else 0
                return httpx.Response(
                    200,
                    json={
                        "room": self.room,
                        "messages": msgs,
                        "latest_seq": latest_seq,
                        "has_more": has_more,
                    },
                )
            if method == "POST":
                body = json.loads(request.content)
                cid = body.get("client_request_id")
                sender_scope = (
                    "user" if body["sender_kind"] == "user" else body["participant_id"]
                )
                if cid:
                    for existing in self.messages:
                        existing_scope = (
                            "user"
                            if existing["sender_kind"] == "user"
                            else existing["participant_id"]
                        )
                        if (
                            existing_scope == sender_scope
                            and existing.get("client_request_id") == cid
                        ):
                            return httpx.Response(
                                201,
                                json={
                                    "message": existing,
                                    "room": self.room,
                                    "effective_handoff_to_participant_ids": existing[
                                        "handoff_to_participant_ids"
                                    ],
                                    "turn_cap_reached": self.room[
                                        "consecutive_agent_turns"
                                    ]
                                    >= self.room["max_consecutive_agent_turns"],
                                    "handoff_suppressed": existing[
                                        "handoff_suppressed"
                                    ],
                                    "idempotent_replay": True,
                                    "enqueued_delivery_ids": [],
                                },
                            )
                if body["sender_kind"] != "user":
                    if (
                        self.room["consecutive_agent_turns"]
                        >= self.room["max_consecutive_agent_turns"]
                    ):
                        return httpx.Response(
                            409,
                            json={
                                "detail": "Agents reached the consecutive turn limit.",
                                "code": "agent_turn_cap_reached",
                            },
                        )
                requested_handoff = list(
                    dict.fromkeys(body.get("handoff_to_participant_ids") or [])
                )
                new_turns = (
                    0
                    if body["sender_kind"] == "user"
                    else self.room["consecutive_agent_turns"] + 1
                )
                cap_reached = new_turns >= self.room["max_consecutive_agent_turns"]
                effective = [] if cap_reached else requested_handoff
                suppressed = cap_reached and bool(requested_handoff)
                seq = (self.messages[-1]["seq"] if self.messages else 0) + 1
                msg = {
                    "id": str(uuid4()),
                    "room_id": str(self.room_id),
                    "seq": seq,
                    "sender_kind": body["sender_kind"],
                    "participant_id": body.get("participant_id"),
                    "sender_name": body["sender_name"],
                    "content": body["content"],
                    "handoff_to_participant_ids": effective,
                    "handoff_suppressed": suppressed,
                    "reply_to_message_id": body.get("reply_to_message_id"),
                    "client_request_id": cid,
                    "created_at": now,
                }
                self.messages.append(msg)
                self.room["consecutive_agent_turns"] = new_turns
                self.room["version"] += 1
                self.room["updated_at"] = now
                return httpx.Response(
                    201,
                    json={
                        "message": msg,
                        "room": self.room,
                        "effective_handoff_to_participant_ids": effective,
                        "turn_cap_reached": cap_reached,
                        "handoff_suppressed": suppressed,
                        "idempotent_replay": False,
                        "enqueued_delivery_ids": [],
                    },
                )

        if path == f"/api/group-chat/rooms/{self.room_id}/subscriptions":
            if method == "GET":
                pid = request.url.params.get("participant_id")
                rows = list(self.subscriptions.values())
                if pid:
                    rows = [r for r in rows if r["participant_id"] == pid]
                return httpx.Response(200, json=rows)
            if method == "POST":
                body = json.loads(request.content)
                existing = next(
                    (
                        item
                        for item in self.subscriptions.values()
                        if item["participant_id"] == body["participant_id"]
                        and item["event_name"] == body["event_name"]
                    ),
                    None,
                )
                if existing is not None:
                    existing.update(
                        {
                            "filters": body.get("filters") or {},
                            "callback_url": body["callback_url"],
                            "signing_secret": body["signing_secret"],
                            "status": "active",
                            "expires_at": body["expires_at"],
                            "last_error": None,
                            "version": existing["version"] + 1,
                            "updated_at": now,
                        }
                    )
                    return httpx.Response(201, json=existing)
                sub = {
                    "id": body.get("id") or str(uuid4()),
                    "room_id": str(self.room_id),
                    "participant_id": body["participant_id"],
                    "event_name": body["event_name"],
                    "filters": body.get("filters") or {},
                    "callback_url": body["callback_url"],
                    "signing_secret": body["signing_secret"],
                    "status": "active",
                    "expires_at": body["expires_at"],
                    "last_delivered_seq": 0,
                    "last_error": None,
                    "version": 1,
                    "created_at": now,
                    "updated_at": now,
                }
                self.subscriptions[sub["id"]] = sub
                return httpx.Response(201, json=sub)

        if path.startswith("/api/group-chat/subscriptions/") and method == "DELETE":
            sub_id = path.removeprefix("/api/group-chat/subscriptions/")
            pid = request.url.params.get("participant_id")
            sub = self.subscriptions.get(sub_id)
            if sub:
                if pid and sub["participant_id"] != pid:
                    return httpx.Response(
                        403,
                        json={"detail": "Subscription belongs to another participant"},
                    )
                sub["status"] = "revoked"
            return httpx.Response(204)

        return httpx.Response(404, json={"detail": f"Unhandled {method} {path}"})


def _make_services(
    tsuchi: FakeTsuchiGroupChat,
    webhook_handler=None,
) -> tuple[LocalServices, httpx.AsyncClient]:
    def router(request: httpx.Request) -> httpx.Response:
        if request.url.host == "tsuchi.test":
            return tsuchi.handle_tsuchi(request)
        if webhook_handler is not None:
            return webhook_handler(request)
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(router))
    settings = Settings(
        tsuchi_url="http://tsuchi.test", hikari_url="http://hikari.test"
    )
    return LocalServices(client, settings), client


def test_mcp_streamable_http_auth_identity_handoff_and_turn_cap():
    tsuchi = FakeTsuchiGroupChat(max_turns=2)
    kadan = tsuchi.add_participant(
        display_name="Kadan", kind="kadan_agent", handle="kadan", agent_key="kadan"
    )
    dot_token = "kdn_gc_dot_live_token_123456"
    dot = tsuchi.add_participant(
        display_name="Dot", kind="external_agent", handle="dot", raw_token=dot_token
    )
    paused_token = "kdn_gc_paused_token_123456"
    tsuchi.add_participant(
        display_name="PausedBot",
        kind="external_agent",
        handle="paused",
        raw_token=paused_token,
        enabled=False,
    )

    app = create_app(
        Settings(tsuchi_url="http://tsuchi.test", hikari_url="http://hikari.test")
    )
    with TestClient(app) as client:
        mock_http = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: tsuchi.handle_tsuchi(req))
        )
        app.state.services = LocalServices(
            mock_http,
            Settings(tsuchi_url="http://tsuchi.test", hikari_url="http://hikari.test"),
        )
        app.state.group_chat_rate_limiter = GroupChatRateLimiter(max_requests=50)

        # GET returns 405 Method Not Allowed cleanly.
        assert client.get("/mcp/group-chat").status_code == 405

        # Missing / invalid / paused bearer tokens rejected.
        assert (
            client.post(
                "/mcp/group-chat",
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/mcp/group-chat",
                headers={"Authorization": "Bearer wrong-token"},
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/mcp/group-chat",
                headers={"Authorization": f"Bearer {paused_token}"},
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            ).status_code
            == 403
        )

        headers = {"Authorization": f"Bearer {dot_token}"}

        # Initialize advertises MCP Events 2026-07-28 and events: {}.
        init = client.post(
            "/mcp/group-chat",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        )
        assert init.status_code == 200
        init_res = init.json()["result"]
        assert init_res["protocolVersion"] == "2026-07-28"
        assert "events" in init_res["capabilities"]
        assert "tools" in init_res["capabilities"]

        # Notification returns 202 Accepted.
        notif = client.post(
            "/mcp/group-chat",
            headers=headers,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        assert notif.status_code == 202

        # tools/list exposes all 4 Group Chat tools.
        tools = client.post(
            "/mcp/group-chat",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ).json()["result"]["tools"]
        assert {t["name"] for t in tools} == {
            "group_chat_list_participants",
            "group_chat_read_messages",
            "group_chat_send_message",
            "group_chat_wait_for_messages",
        }

        # Self-handoff is rejected.
        self_call = client.post(
            "/mcp/group-chat",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "group_chat_send_message",
                    "arguments": {"content": "Ping myself", "handoff_to": "@dot"},
                },
            },
        ).json()["result"]
        assert self_call["isError"] is True
        assert self_call["structuredContent"]["code"] == "self_handoff_not_allowed"

        # Turn 1: Dot hands off to Kadan -> identity bound server-side to Dot, wakes Kadan!
        send_1 = client.post(
            "/mcp/group-chat",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "group_chat_send_message",
                    "arguments": {
                        "content": "Over to @kadan",
                        "handoff_to": "kadan",
                        "client_request_id": "dot-msg-1",
                    },
                },
            },
        ).json()["result"]
        assert send_1["isError"] is False
        assert send_1["structuredContent"]["sender_name"] == "Dot"
        assert send_1["structuredContent"]["sender_kind"] == "external_agent"
        assert send_1["structuredContent"]["seq"] == 1
        assert send_1["structuredContent"]["turn_cap_reached"] is False
        assert len(tsuchi.event_runs) == 1
        assert tsuchi.event_runs[0]["trigger_event"]["kind"] == "group_chat"

        # Turn 2: hits turn cap (2/2), handoff suppressed.
        send_2 = client.post(
            "/mcp/group-chat",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "group_chat_send_message",
                    "arguments": {
                        "content": "Final turn",
                        "handoff_to": kadan["id"],
                        "client_request_id": "dot-msg-2",
                    },
                },
            },
        ).json()["result"]
        assert send_2["isError"] is False
        assert send_2["structuredContent"]["turn_cap_reached"] is True
        assert send_2["structuredContent"]["handoff_suppressed"] is True
        assert send_2["structuredContent"]["handoff_to"] == []
        # No second wake-up was queued on the cap-hitting turn.
        assert len(tsuchi.event_runs) == 1

        # Turn 3: rejected with agent_turn_cap_reached for external agent.
        send_3 = client.post(
            "/mcp/group-chat",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {
                    "name": "group_chat_send_message",
                    "arguments": {
                        "content": "Too many turns",
                        "client_request_id": "dot-msg-3",
                    },
                },
            },
        ).json()["result"]
        assert send_3["isError"] is True
        assert send_3["structuredContent"]["code"] == "agent_turn_cap_reached"
        assert send_3["structuredContent"]["turn_cap_reached"] is True

        # group_chat_read_messages reports addressed_to_me and turn-cap status.
        read_res = client.post(
            "/mcp/group-chat",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {
                    "name": "group_chat_read_messages",
                    "arguments": {"after_seq": 0, "limit": 10},
                },
            },
        ).json()["result"]
        assert read_res["isError"] is False
        assert read_res["structuredContent"]["turn_cap_reached"] is True
        assert len(read_res["structuredContent"]["messages"]) == 2
        assert dot["id"] in {p["id"] for p in tsuchi.participants.values()}


def test_mcp_events_list_subscribe_challenge_and_unsubscribe():
    tsuchi = FakeTsuchiGroupChat()
    dot_token = "kdn_gc_dot_events_token_123"
    dot = tsuchi.add_participant(
        display_name="Dot", kind="external_agent", handle="dot", raw_token=dot_token
    )
    raw_secret = b"0123456789abcdef0123456789abcdef"
    signing_secret = "whsec_" + base64.b64encode(raw_secret).decode("ascii")

    challenge_calls: list[dict[str, Any]] = []
    mode = {"behavior": "echo"}

    def webhook_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        challenge_calls.append({"headers": dict(request.headers), "body": body})
        if mode["behavior"] == "redirect":
            return httpx.Response(
                302, headers={"Location": "https://other.example/hook"}
            )
        if mode["behavior"] == "mismatch":
            return httpx.Response(200, json={"challenge": "wrong-challenge"})
        # Verify Standard Webhooks HMAC signature on the challenge itself!
        wh_id = request.headers["webhook-id"]
        wh_ts = request.headers["webhook-timestamp"]
        expected = "v1," + base64.b64encode(
            hmac.new(
                raw_secret,
                f"{wh_id}.{wh_ts}.".encode("utf-8") + request.content,
                hashlib.sha256,
            ).digest()
        ).decode("ascii")
        assert request.headers["webhook-signature"] == expected
        assert request.headers["x-mcp-subscription-id"] == body["subscriptionId"]
        return httpx.Response(200, json={"challenge": body["challenge"]})

    async def scenario():
        services, client = _make_services(tsuchi, webhook_handler=webhook_handler)
        try:
            handler = GroupChatMcpHandler(services)
            participant = await services.group_chat.authenticate_token(dot_token)

            # events/list advertises group_chat.turn_requested.
            status, body = await handler.handle_jsonrpc(
                participant, {"jsonrpc": "2.0", "id": 1, "method": "events/list"}
            )
            assert status == 200
            assert body is not None
            assert body["result"]["events"][0]["name"] == "group_chat.turn_requested"

            # Redirect during challenge is rejected without persisting subscription.
            mode["behavior"] = "redirect"
            _, redir = await handler.handle_jsonrpc(
                participant,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "events/subscribe",
                    "params": {
                        "event": "group_chat.turn_requested",
                        "webhook": {
                            "url": "https://dot.example/events",
                            "secret": signing_secret,
                        },
                        "ttlSeconds": 1800,
                    },
                },
            )
            assert redir is not None and "error" in redir
            assert redir["error"]["data"]["code"] == "challenge_redirect_rejected"
            assert tsuchi.subscriptions == {}

            # Challenge mismatch is rejected.
            mode["behavior"] = "mismatch"
            _, mismatch = await handler.handle_jsonrpc(
                participant,
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "events/subscribe",
                    "params": {
                        "event": "group_chat.turn_requested",
                        "webhook": {
                            "url": "https://dot.example/events",
                            "secret": signing_secret,
                        },
                    },
                },
            )
            assert mismatch is not None and "error" in mismatch
            assert mismatch["error"]["data"]["code"] == "challenge_mismatch"
            assert tsuchi.subscriptions == {}

            # Valid challenge succeeds and persists subscription (OpenAI MCP Events contract: id + refreshBefore).
            mode["behavior"] = "echo"
            _, sub_res = await handler.handle_jsonrpc(
                participant,
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "events/subscribe",
                    "params": {
                        "name": "group_chat.turn_requested",
                        "arguments": {"room_id": str(tsuchi.room_id)},
                        "webhook": {
                            "url": "https://dot.example/events",
                            "secret": signing_secret,
                        },
                        "ttlSeconds": 1800,
                    },
                },
            )
            assert sub_res is not None and "result" in sub_res
            result = sub_res["result"]
            assert "id" in result and "refreshBefore" in result
            assert "subscriptionId" not in result and "expiresAt" not in result
            sub_id = result["id"]
            assert sub_id in tsuchi.subscriptions
            assert tsuchi.subscriptions[sub_id]["participant_id"] == dot["id"]
            assert tsuchi.subscriptions[sub_id]["filters"] == {
                "room_id": str(tsuchi.room_id)
            }

            # Browser public participant projection NEVER exposes signing_secret!
            public_participants = await services.group_chat.list_public_participants(
                tsuchi.room_id
            )
            dot_pub = next(p for p in public_participants if p.id == UUID(dot["id"]))
            assert dot_pub.push_status == "active"
            assert signing_secret not in dot_pub.model_dump_json()

            # events/unsubscribe accepts `id` and idempotently revokes the subscription.
            _, unsub = await handler.handle_jsonrpc(
                participant,
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "events/unsubscribe",
                    "params": {"id": sub_id},
                },
            )
            assert unsub is not None
            assert unsub["result"] == {"id": sub_id, "unsubscribed": True}
            assert tsuchi.subscriptions[sub_id]["status"] == "revoked"
        finally:
            await client.aclose()

    asyncio.run(scenario())


def test_mcp_events_subscribe_refresh_reuses_id_and_unsubscribe_contract():
    tsuchi = FakeTsuchiGroupChat()
    dot_token = "kdn_gc_dot_contract_token_456"
    tsuchi.add_participant(
        display_name="Dot", kind="external_agent", handle="dot", raw_token=dot_token
    )
    raw_secret = b"fedcba9876543210fedcba9876543210"
    signing_secret = "whsec_" + base64.b64encode(raw_secret).decode("ascii")
    seen_challenge_sub_ids: list[str] = []

    def webhook_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen_challenge_sub_ids.append(request.headers["x-mcp-subscription-id"])
        return httpx.Response(200, json={"challenge": body["challenge"]})

    async def scenario():
        services, client = _make_services(tsuchi, webhook_handler=webhook_handler)
        try:
            handler = GroupChatMcpHandler(services)
            participant = await services.group_chat.authenticate_token(dot_token)

            # events/list advertises `arguments` schema matching events/subscribe.
            _, listed = await handler.handle_jsonrpc(
                participant, {"jsonrpc": "2.0", "id": 1, "method": "events/list"}
            )
            assert listed is not None
            event_spec = listed["result"]["events"][0]
            assert "arguments" in event_spec
            assert "room_id" in event_spec["arguments"]["properties"]

            # 1. Initial subscribe with `arguments` returns `id` and `refreshBefore`.
            _, sub1 = await handler.handle_jsonrpc(
                participant,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "events/subscribe",
                    "params": {
                        "name": "group_chat.turn_requested",
                        "arguments": {"room_id": str(tsuchi.room_id)},
                        "webhook": {
                            "url": "https://dot.example/events",
                            "secret": signing_secret,
                        },
                        "ttlSeconds": 600,
                    },
                },
            )
            assert sub1 is not None and "result" in sub1
            r1 = sub1["result"]
            assert set(r1) == {"id", "event", "status", "refreshBefore"}
            first_id = r1["id"]
            first_refresh = datetime.fromisoformat(r1["refreshBefore"])
            assert first_refresh > datetime.now(UTC)

            # 2. Refreshing reuses the exact same durable `id` and extends `refreshBefore`.
            _, sub2 = await handler.handle_jsonrpc(
                participant,
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "events/subscribe",
                    "params": {
                        "name": "group_chat.turn_requested",
                        "arguments": {"room_id": str(tsuchi.room_id)},
                        "webhook": {
                            "url": "https://dot.example/events-v2",
                            "secret": signing_secret,
                        },
                        "ttlSeconds": 3600,
                    },
                },
            )
            assert sub2 is not None and "result" in sub2
            r2 = sub2["result"]
            assert r2["id"] == first_id
            assert datetime.fromisoformat(r2["refreshBefore"]) > first_refresh
            assert len(tsuchi.subscriptions) == 1
            assert tsuchi.subscriptions[first_id]["version"] == 2
            assert (
                tsuchi.subscriptions[first_id]["callback_url"]
                == "https://dot.example/events-v2"
            )
            assert seen_challenge_sub_ids == [first_id, first_id]

            # 3. Unsubscribe with `id` revokes the durable subscription.
            _, unsub = await handler.handle_jsonrpc(
                participant,
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "events/unsubscribe",
                    "params": {"id": first_id},
                },
            )
            assert unsub is not None
            assert unsub["result"] == {"id": first_id, "unsubscribed": True}
            assert tsuchi.subscriptions[first_id]["status"] == "revoked"
        finally:
            await client.aclose()

    asyncio.run(scenario())


def test_wait_for_messages_fallback_respects_addressed_and_turn_cap():
    tsuchi = FakeTsuchiGroupChat(max_turns=2)
    kadan = tsuchi.add_participant(
        display_name="Kadan", kind="kadan_agent", handle="kadan", agent_key="kadan"
    )
    dot_token = "kdn_gc_dot_wait_token_123"
    dot = tsuchi.add_participant(
        display_name="Dot", kind="external_agent", handle="dot", raw_token=dot_token
    )

    async def scenario():
        services, client = _make_services(tsuchi)
        try:
            handler = GroupChatMcpHandler(services, wait_poll_interval=0.02)
            participant = await services.group_chat.authenticate_token(dot_token)

            # Untagged message does NOT wake default only_when_addressed=True wait.
            await services.group_chat.create_message(
                tsuchi.room_id,
                sender_kind="user",
                sender_name="You",
                content="Just thinking out loud",
                handoff_to_participant_ids=[],
            )
            res_untagged, is_err = await handler.call_tool(
                participant,
                "group_chat_wait_for_messages",
                {"after_seq": 0, "timeout_seconds": 0.08},
            )
            assert not is_err
            assert res_untagged["timed_out"] is True
            assert res_untagged["messages"] == []

            # Message addressed to Dot wakes immediately and returns ALL unread messages since after_seq!
            await services.group_chat.create_message(
                tsuchi.room_id,
                sender_kind="user",
                sender_name="You",
                content="Hey @dot please weigh in",
                handoff_to_participant_ids=[UUID(dot["id"])],
            )
            res_addressed, is_err = await handler.call_tool(
                participant,
                "group_chat_wait_for_messages",
                {"after_seq": 0, "timeout_seconds": 0.2},
            )
            assert not is_err
            assert res_addressed["timed_out"] is False
            assert [m["seq"] for m in res_addressed["messages"]] == [1, 2]
            assert res_addressed["trigger_seq"] == 2
            assert res_addressed["messages"][1]["addressed_to_me"] is True

            # When room is at turn cap, wait_for_messages stays quiet until user resets it.
            tsuchi.room["consecutive_agent_turns"] = 2
            tsuchi.messages.append(
                {
                    "id": str(uuid4()),
                    "room_id": str(tsuchi.room_id),
                    "seq": 3,
                    "sender_kind": "kadan_agent",
                    "participant_id": kadan["id"],
                    "sender_name": "Kadan",
                    "content": "At cap",
                    "handoff_to_participant_ids": [dot["id"]],
                    "handoff_suppressed": False,
                    "reply_to_message_id": None,
                    "client_request_id": None,
                    "created_at": (
                        datetime.now(UTC) + timedelta(seconds=1)
                    ).isoformat(),
                }
            )
            res_capped, is_err = await handler.call_tool(
                participant,
                "group_chat_wait_for_messages",
                {"after_seq": 2, "timeout_seconds": 0.08},
            )
            assert not is_err
            assert res_capped["timed_out"] is True
            assert res_capped["turn_cap_reached"] is True
        finally:
            await client.aclose()

    asyncio.run(scenario())


def test_browser_group_chat_routes_create_rotate_revoke_and_redact_secrets():
    tsuchi = FakeTsuchiGroupChat()
    app = create_app(
        Settings(tsuchi_url="http://tsuchi.test", hikari_url="http://hikari.test")
    )
    with TestClient(app) as client:
        mock_http = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: tsuchi.handle_tsuchi(req))
        )
        app.state.services = LocalServices(
            mock_http,
            Settings(tsuchi_url="http://tsuchi.test", hikari_url="http://hikari.test"),
        )

        # Snapshot ensures Kadan's built-in participant and exposes copyable mcp_url.
        snap = client.get("/api/group-chat").json()
        assert snap["room"]["title"] == "Group Chat"
        assert snap["mcp_url"].endswith("/mcp/group-chat")
        assert any(p["kind"] == "kadan_agent" for p in snap["participants"])

        # Create external participant returns one-time bearer token.
        created = client.post(
            "/api/group-chat/participants",
            json={"display_name": "Dot", "description": "Code reviewer"},
        )
        assert created.status_code == 201
        created_body = created.json()
        token1 = created_body["bearer_token"]
        assert token1.startswith("kdn_gc_")
        dot_id = created_body["participant"]["id"]

        # Subsequent GET snapshot never contains the raw token or signing secrets.
        snap2 = client.get("/api/group-chat").text
        assert token1 not in snap2

        # Rotate token returns a new one-time bearer token.
        rotated = client.put(
            f"/api/group-chat/participants/{dot_id}",
            json={"rotate_token": True, "expected_version": 1},
        )
        assert rotated.status_code == 200
        token2 = rotated.json()["bearer_token"]
        assert token2 and token2 != token1

        # Posting a user message targeting Kadan wakes Kadan.
        kadan_id = next(
            p["id"] for p in snap["participants"] if p["kind"] == "kadan_agent"
        )
        posted = client.post(
            "/api/group-chat/messages",
            json={
                "content": "Hi @kadan",
                "handoff_to_participant_ids": [kadan_id],
                "client_request_id": "browser-1",
            },
        )
        assert posted.status_code == 201
        assert posted.json()["message"]["seq"] == 1
        assert len(tsuchi.event_runs) == 1
