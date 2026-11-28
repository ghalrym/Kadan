"""Unit tests for private 1-on-1 chats between the user and one external Group Chat agent."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4, uuid5

import httpx
from fastapi.testclient import TestClient

from kadan.api.config import Settings
from kadan.api.group_chat_mcp import GroupChatMcpHandler
from kadan.api.pydantic_models.group_chat import GroupChatParticipant
from kadan.api.server import create_app
from kadan.api.services import LocalServices

SETTINGS = Settings(tsuchi_url="http://tsuchi.test", hikari_url="http://hikari.test")
HOME = UUID("00000000-0000-0000-0000-000000000001")


def _room(room_id: UUID, title: str, **extra: Any) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    return {
        "id": str(room_id),
        "title": title,
        "consecutive_agent_turns": 0,
        "max_consecutive_agent_turns": 6,
        "version": 1,
        "created_at": now,
        "updated_at": now,
        **extra,
    }


def _participant(name: str, kind: str, **extra: Any) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    return {
        "id": str(uuid4()),
        "room_id": str(HOME),
        "handle": name.lower(),
        "display_name": name,
        "kind": kind,
        "agent_key": "kadan" if kind == "kadan_agent" else None,
        "description": "",
        "enabled": True,
        "revoked_at": None,
        "last_seen_at": None,
        "cursor_seq": 0,
        "version": 1,
        "created_at": now,
        "updated_at": now,
        **extra,
    }


class FakeTsuchi:
    """Just enough Tsuchi for direct rooms: rooms, messages, roster, profile and touches."""

    def __init__(self) -> None:
        self.kadan = _participant("Kadan", "kadan_agent")
        self.dot = _participant("Dot", "external_agent")
        self.sol = _participant("Sol", "external_agent")
        self.paused = _participant("Nap", "external_agent", enabled=False)
        self.participants = [self.kadan, self.dot, self.sol, self.paused]
        self.rooms: dict[str, dict[str, Any]] = {str(HOME): _room(HOME, "Group Chat")}
        self.messages: dict[str, list[dict[str, Any]]] = {str(HOME): []}
        self.touches: list[dict[str, Any]] = []

    def direct(self, participant_id: str) -> dict[str, Any]:
        room_id = uuid5(UUID(participant_id), "kadan-direct-room")
        if str(room_id) not in self.rooms:
            self.rooms[str(room_id)] = _room(
                room_id, "Direct", kind="direct", direct_participant_id=participant_id
            )
            self.messages[str(room_id)] = []
        return self.rooms[str(room_id)]

    def handle(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if path == "/api/profile":
            return httpx.Response(200, json={"display_name": "Andrew", "pet": None, "version": 1})
        if path == "/api/group-chat/rooms/default":
            return httpx.Response(200, json=self.rooms[str(HOME)])
        if path == "/api/group-chat/direct-rooms":
            return httpx.Response(
                200,
                json=[
                    {
                        "room": room,
                        "latest_seq": len(self.messages[room_id]),
                        "last_message_at": (
                            self.messages[room_id][-1]["created_at"]
                            if self.messages[room_id]
                            else None
                        ),
                    }
                    for room_id, room in self.rooms.items()
                    if room.get("kind") == "direct"
                ],
            )
        if path.endswith("/direct-room") and method == "POST":
            return httpx.Response(200, json=self.direct(path.split("/")[-2]))
        if path == f"/api/group-chat/rooms/{HOME}/participants":
            return httpx.Response(200, json=self.participants)
        if path.endswith("/touch"):
            self.touches.append(json.loads(request.content))
            return httpx.Response(200, json=self.dot)
        if path.startswith("/api/group-chat/rooms/") and path.endswith("/participants"):
            return httpx.Response(200, json=[])
        if path.startswith("/api/group-chat/rooms/") and path.endswith("/messages"):
            room_id = path.split("/")[-2]
            room = self.rooms[room_id]
            messages = self.messages[room_id]
            if method == "GET":
                after = request.url.params.get("after_seq")
                rows = [m for m in messages if after is None or m["seq"] > int(after)]
                return httpx.Response(
                    200,
                    json={
                        "room": room,
                        "messages": rows,
                        "latest_seq": len(messages),
                        "has_more": False,
                    },
                )
            body = json.loads(request.content)
            message = {
                "id": str(uuid4()),
                "room_id": room_id,
                "seq": len(messages) + 1,
                "sender_kind": body["sender_kind"],
                "participant_id": body.get("participant_id"),
                "sender_name": body["sender_name"],
                "content": body["content"],
                "handoff_to_participant_ids": body.get("handoff_to_participant_ids") or [],
                "handoff_suppressed": False,
                "reply_to_message_id": body.get("reply_to_message_id"),
                "client_request_id": body.get("client_request_id"),
                "created_at": datetime.now(UTC).isoformat(),
            }
            messages.append(message)
            return httpx.Response(
                201,
                json={
                    "message": message,
                    "room": room,
                    "effective_handoff_to_participant_ids": message["handoff_to_participant_ids"],
                    "turn_cap_reached": False,
                    "handoff_suppressed": False,
                    "idempotent_replay": False,
                    "enqueued_delivery_ids": [],
                },
            )
        if path.startswith("/api/group-chat/rooms/") and path.endswith("/subscriptions"):
            return httpx.Response(200, json=[])
        if path.startswith("/api/group-chat/rooms/"):
            return httpx.Response(200, json=self.rooms[path.split("/")[-1]])
        raise AssertionError(f"Unexpected Tsuchi call: {method} {path}")


def _services(tsuchi: FakeTsuchi) -> LocalServices:
    return LocalServices(
        httpx.AsyncClient(transport=httpx.MockTransport(tsuchi.handle)), SETTINGS
    )


def test_browser_direct_chats_list_open_and_always_ask_the_agent_to_reply():
    tsuchi = FakeTsuchi()
    app = create_app(SETTINGS)
    with TestClient(app) as client:
        app.state.services = _services(tsuchi)

        listed = client.get("/api/group-chat/direct").json()
        assert [item["participant"]["display_name"] for item in listed] == ["Dot", "Sol"]
        assert all(item["room_id"] is None and item["latest_seq"] == 0 for item in listed)

        snapshot = client.get(f"/api/group-chat/direct/{tsuchi.dot['id']}").json()
        assert snapshot["room"]["kind"] == "direct"
        assert [item["display_name"] for item in snapshot["participants"]] == ["Dot"]
        assert snapshot["messages"] == []

        posted = client.post(
            f"/api/group-chat/direct/{tsuchi.dot['id']}/messages",
            json={"content": "Just us", "client_request_id": "dm-1"},
        )
        assert posted.status_code == 201
        message = posted.json()["message"]
        assert message["handoff_to_participant_ids"] == [tsuchi.dot["id"]]
        assert message["sender_name"] == "Andrew"
        assert tsuchi.messages[str(HOME)] == []

        listed = client.get("/api/group-chat/direct").json()
        dot_entry = next(item for item in listed if item["participant"]["id"] == tsuchi.dot["id"])
        assert dot_entry["latest_seq"] == 1
        assert dot_entry["room_id"] == snapshot["room"]["id"]

        assert client.get(f"/api/group-chat/direct/{tsuchi.kadan['id']}").status_code == 404
        assert (
            client.post(
                f"/api/group-chat/direct/{tsuchi.paused['id']}/messages",
                json={"content": "Wake up"},
            ).status_code
            == 409
        )


def test_mcp_room_argument_reads_and_replies_in_the_direct_room_without_moving_the_cursor():
    tsuchi = FakeTsuchi()

    async def scenario() -> None:
        services = _services(tsuchi)
        handler = GroupChatMcpHandler(services)
        dot = GroupChatParticipant.model_validate(tsuchi.dot)
        kadan = GroupChatParticipant.model_validate(tsuchi.kadan)
        direct = tsuchi.direct(tsuchi.dot["id"])
        tsuchi.messages[direct["id"]].append(
            {
                "id": str(uuid4()),
                "room_id": direct["id"],
                "seq": 1,
                "sender_kind": "user",
                "participant_id": None,
                "sender_name": "Andrew",
                "content": "Private question",
                "handoff_to_participant_ids": [tsuchi.dot["id"]],
                "handoff_suppressed": False,
                "reply_to_message_id": None,
                "client_request_id": None,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )

        read, is_error = await handler.call_tool(dot, "group_chat_read_messages", {"room": "direct"})
        assert not is_error
        assert read["room"] == "direct"
        assert [m["content"] for m in read["messages"]] == ["Private question"]
        assert read["messages"][0]["addressed_to_me"] is True

        sent, is_error = await handler.call_tool(
            dot,
            "group_chat_send_message",
            {"room": "direct", "content": "Private answer", "client_request_id": "a1"},
        )
        assert not is_error
        assert sent["room"] == "direct"
        assert tsuchi.messages[direct["id"]][-1]["content"] == "Private answer"
        assert tsuchi.messages[str(HOME)] == []
        assert tsuchi.touches and all("cursor_seq" not in touch for touch in tsuchi.touches)

        roster, _ = await handler.call_tool(dot, "group_chat_list_participants", {"room": "direct"})
        assert [p["handle"] for p in roster["participants"]] == ["user", "dot"]

        group, _ = await handler.call_tool(dot, "group_chat_read_messages", {})
        assert group["room"] == "group"

        denied, is_error = await handler.call_tool(kadan, "group_chat_read_messages", {"room": "direct"})
        assert is_error and denied["code"] == "invalid_arguments"
        bad, is_error = await handler.call_tool(dot, "group_chat_read_messages", {"room": "elsewhere"})
        assert is_error and bad["code"] == "invalid_arguments"

    asyncio.run(scenario())
