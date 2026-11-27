"""Streamable HTTP MCP server & MCP Events (SEP-3415) handler for Kadan Group Chat."""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4, uuid5

import httpx

from .config import ServiceError
from .group_chat_webhooks import (
    build_webhook_headers,
    decode_whsec_secret,
    validate_webhook_callback_url,
)
from .pydantic_models.group_chat import (
    GroupChatMessage,
    GroupChatParticipant,
    GroupChatRoom,
)

if TYPE_CHECKING:
    from .services import LocalServices

MCP_EVENTS_PROTOCOL_VERSION = "2026-07-28"
EVENT_TURN_REQUESTED = "group_chat.turn_requested"
DEFAULT_SUBSCRIPTION_TTL_SECONDS = 3600
MIN_SUBSCRIPTION_TTL_SECONDS = 60
MAX_SUBSCRIPTION_TTL_SECONDS = 86400
MAX_WAIT_TIMEOUT_SECONDS = 30
DEFAULT_WAIT_TIMEOUT_SECONDS = 20
DEFAULT_WAIT_POLL_INTERVAL = 0.5
CHALLENGE_TIMEOUT_SECONDS = 10.0

GROUP_CHAT_MCP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "group_chat_list_participants",
        "description": (
            "List participants in the Kadan Group Chat room, including the user, "
            "Kadan's built-in agent, and external agents with their @handles and status."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "annotations": {
            "readOnlyHint": True,
            "idempotentHint": True,
        },
    },
    {
        "name": "group_chat_read_messages",
        "description": (
            "Read paginated messages from the Kadan Group Chat room using monotonic "
            "sequence numbers (after_seq / before_seq), including turn-cap status."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "after_seq": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Return messages strictly newer than this sequence number.",
                },
                "before_seq": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Return messages strictly older than this sequence number.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 50,
                    "description": "Maximum number of messages to return (1-100).",
                },
            },
            "additionalProperties": False,
        },
        "annotations": {
            "readOnlyHint": True,
            "idempotentHint": True,
        },
    },
    {
        "name": "group_chat_send_message",
        "description": (
            "Post a message to the Kadan Group Chat room as the authenticated participant. "
            "Optionally pass handoff_to to explicitly hand the turn to another agent, "
            "or omit handoff_to to return the floor to the user."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 16000,
                    "description": "Message text to post in Group Chat.",
                },
                "handoff_to": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}, "maxItems": 10},
                        {"type": "null"},
                    ],
                    "description": (
                        "Optional participant @handle or UUID (or list of them) to explicitly "
                        "hand off the turn to. Self-handoff is rejected. Omit to end the relay."
                    ),
                },
                "reply_to_message_id": {
                    "type": ["string", "null"],
                    "description": "Optional UUID of the message being replied to.",
                },
                "client_request_id": {
                    "type": ["string", "null"],
                    "maxLength": 256,
                    "description": "Optional idempotency key so retries never double-post.",
                },
            },
            "required": ["content"],
            "additionalProperties": False,
        },
        "annotations": {
            "readOnlyHint": False,
            "idempotentHint": True,
        },
    },
    {
        "name": "group_chat_wait_for_messages",
        "description": (
            "Bounded long-poll fallback for clients without MCP Events push webhooks. "
            "Defaults to only_when_addressed=true and stays quiet when the room is at the turn cap."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "after_seq": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Sequence number to wait after (defaults to your last cursor).",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_WAIT_TIMEOUT_SECONDS,
                    "default": DEFAULT_WAIT_TIMEOUT_SECONDS,
                    "description": "Maximum seconds to wait before returning (1-30).",
                },
                "only_when_addressed": {
                    "type": "boolean",
                    "default": True,
                    "description": "When true (default), wake only when a message tags or hands off to you.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 50,
                    "description": "Maximum unread messages to return (1-100).",
                },
            },
            "additionalProperties": False,
        },
        "annotations": {
            "readOnlyHint": True,
            "idempotentHint": True,
        },
    },
]

GROUP_CHAT_MCP_EVENTS: list[dict[str, Any]] = [
    {
        "name": EVENT_TURN_REQUESTED,
        "description": (
            "Fired when this participant is explicitly tagged by the user or handed "
            "the turn by another agent in Group Chat (and the room has not hit its turn cap)."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "room_id": {"type": "string"},
                "room_title": {"type": "string"},
                "seq": {"type": "integer"},
                "message": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "seq": {"type": "integer"},
                        "sender_kind": {"type": "string"},
                        "sender_name": {"type": "string"},
                        "content": {"type": "string"},
                        "handoff_to_participant_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "created_at": {"type": "string"},
                    },
                },
                "recent_messages": {"type": "array", "items": {"type": "object"}},
                "consecutive_agent_turns": {"type": "integer"},
                "max_consecutive_agent_turns": {"type": "integer"},
                "turn_cap_reached": {"type": "boolean"},
            },
        },
        "arguments": {
            "type": "object",
            "properties": {
                "room_id": {
                    "type": "string",
                    "description": "Optional Group Chat room UUID filter.",
                }
            },
            "additionalProperties": False,
        },
        "delivery": {
            "webhook": {
                "supported": True,
                "signing": "standard-webhooks-hmac-sha256",
            }
        },
    }
]


class GroupChatRateLimiter:
    """Basic per-participant sliding-window request limiter and concurrent long-poll cap."""

    def __init__(
        self,
        *,
        max_requests: int = 120,
        window_seconds: float = 60.0,
        max_concurrent_waits: int = 2,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_concurrent_waits = max_concurrent_waits
        self._requests: dict[UUID, deque[float]] = {}
        self._active_waits: dict[UUID, int] = {}

    def check_request(self, participant_id: UUID) -> tuple[bool, int]:
        now = time.monotonic()
        bucket = self._requests.setdefault(participant_id, deque())
        cutoff = now - self.window_seconds
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= self.max_requests:
            retry_after = max(1, int(bucket[0] + self.window_seconds - now) + 1)
            return False, retry_after
        bucket.append(now)
        return True, 0

    @asynccontextmanager
    async def wait_slot(self, participant_id: UUID):
        current = self._active_waits.get(participant_id, 0)
        if current >= self.max_concurrent_waits:
            raise ServiceError(
                "Too many concurrent wait requests for this participant.",
                429,
                "too_many_waits",
            )
        self._active_waits[participant_id] = current + 1
        try:
            yield
        finally:
            remaining = self._active_waits.get(participant_id, 1) - 1
            if remaining <= 0:
                self._active_waits.pop(participant_id, None)
            else:
                self._active_waits[participant_id] = remaining


class GroupChatMcpHandler:
    """Unified Group Chat MCP tool + MCP Events handler shared by HTTP MCP and Kadan's agent."""

    def __init__(
        self,
        services: LocalServices,
        *,
        rate_limiter: GroupChatRateLimiter | None = None,
        wait_poll_interval: float = DEFAULT_WAIT_POLL_INTERVAL,
    ):
        self.services = services
        self.rate_limiter = rate_limiter or GroupChatRateLimiter()
        self.wait_poll_interval = wait_poll_interval

    @staticmethod
    def _rpc_result(req_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _rpc_error(
        req_id: Any, code: int, message: str, data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": req_id, "error": err}

    @staticmethod
    def _tool_result(
        payload: dict[str, Any], *, is_error: bool = False
    ) -> dict[str, Any]:
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload, ensure_ascii=False),
                }
            ],
            "structuredContent": payload,
            "isError": is_error,
        }

    async def handle_jsonrpc(
        self, participant: GroupChatParticipant, payload: Any
    ) -> tuple[int, dict[str, Any] | None]:
        """Process one JSON-RPC 2.0 request/notification for an authenticated participant."""
        if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0":
            return 400, self._rpc_error(None, -32600, "Invalid JSON-RPC 2.0 request.")
        method = payload.get("method")
        if not isinstance(method, str) or not method.strip():
            return 400, self._rpc_error(
                payload.get("id"), -32600, "Missing JSON-RPC method."
            )
        has_id = "id" in payload
        req_id = payload.get("id")
        params = payload.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return 200, self._rpc_error(
                req_id, -32602, "JSON-RPC params must be an object."
            )

        # Notifications have no id and return 202 Accepted with no body.
        if not has_id or method.startswith("notifications/"):
            return 202, None

        if method == "initialize":
            return 200, self._rpc_result(
                req_id,
                {
                    "protocolVersion": MCP_EVENTS_PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {},
                        "events": {},
                    },
                    "serverInfo": {
                        "name": "kadan-group-chat",
                        "version": "1.0.0",
                    },
                    "instructions": (
                        "You are connected to Kadan Group Chat. Use group_chat_read_messages "
                        "to catch up, group_chat_send_message to reply (pass handoff_to only "
                        "to hand the turn to another agent), and events/subscribe for push wake-ups."
                    ),
                },
            )

        if method == "ping":
            return 200, self._rpc_result(req_id, {})

        if method == "tools/list":
            return 200, self._rpc_result(req_id, {"tools": GROUP_CHAT_MCP_TOOLS})

        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(name, str) or not isinstance(arguments, dict):
                return 200, self._rpc_error(
                    req_id,
                    -32602,
                    "tools/call requires a tool name and arguments object.",
                )
            result, is_error = await self.call_tool(participant, name, arguments)
            return 200, self._rpc_result(
                req_id, self._tool_result(result, is_error=is_error)
            )

        if method == "events/list":
            return 200, self._rpc_result(req_id, {"events": GROUP_CHAT_MCP_EVENTS})

        if method == "events/subscribe":
            try:
                sub_result = await self.subscribe_events(participant, params)
            except ServiceError as error:
                rpc_code = -32602 if error.status in {400, 422} else -32000
                return 200, self._rpc_error(
                    req_id, rpc_code, str(error), {"code": error.code}
                )
            return 200, self._rpc_result(req_id, sub_result)

        if method == "events/unsubscribe":
            try:
                unsub_result = await self.unsubscribe_events(participant, params)
            except ServiceError as error:
                rpc_code = -32602 if error.status in {400, 422} else -32000
                return 200, self._rpc_error(
                    req_id, rpc_code, str(error), {"code": error.code}
                )
            return 200, self._rpc_result(req_id, unsub_result)

        return 200, self._rpc_error(req_id, -32601, f"Method '{method}' not found.")

    async def call_tool(
        self,
        participant: GroupChatParticipant,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        """Dispatch a Group Chat MCP tool call for either an external or Kadan participant."""
        try:
            if name == "group_chat_list_participants":
                return await self.list_participants(participant), False
            if name == "group_chat_read_messages":
                return await self.read_messages(participant, arguments), False
            if name == "group_chat_send_message":
                return await self.send_message(participant, arguments), False
            if name == "group_chat_wait_for_messages":
                return await self.wait_for_messages(participant, arguments), False
            return {"error": f"Unknown tool '{name}'.", "code": "unknown_tool"}, True
        except ServiceError as error:
            payload: dict[str, Any] = {"error": str(error), "code": error.code}
            if error.code == "agent_turn_cap_reached":
                payload["turn_cap_reached"] = True
            return payload, True

    async def list_participants(
        self, participant: GroupChatParticipant
    ) -> dict[str, Any]:
        room = await self.services.group_chat.get_room(participant.room_id)
        participants = await self.services.group_chat.list_participants(room.id)
        entries: list[dict[str, Any]] = [
            {
                "id": None,
                "handle": "user",
                "display_name": "User",
                "kind": "user",
                "description": "The human user in Group Chat",
                "active": True,
                "last_seen_at": None,
            }
        ]
        for item in participants:
            entries.append(
                {
                    "id": str(item.id),
                    "handle": item.handle,
                    "display_name": item.display_name,
                    "kind": item.kind,
                    "description": item.description,
                    "active": item.enabled and item.revoked_at is None,
                    "is_me": item.id == participant.id,
                    "last_seen_at": (
                        item.last_seen_at.isoformat() if item.last_seen_at else None
                    ),
                }
            )
        return {
            "room_id": str(room.id),
            "room_title": room.title,
            "participants": entries,
        }

    @staticmethod
    def format_message(
        message: GroupChatMessage,
        *,
        participant: GroupChatParticipant,
        participants_by_id: dict[UUID, GroupChatParticipant],
    ) -> dict[str, Any]:
        handoff_entries = []
        for target_id in message.handoff_to_participant_ids:
            target = participants_by_id.get(target_id)
            handoff_entries.append(
                {
                    "id": str(target_id),
                    "handle": target.handle if target else str(target_id),
                    "display_name": target.display_name if target else str(target_id),
                }
            )
        return {
            "seq": message.seq,
            "id": str(message.id),
            "sender_kind": message.sender_kind,
            "participant_id": (
                str(message.participant_id) if message.participant_id else None
            ),
            "sender_name": message.sender_name,
            "content": message.content,
            "handoff_to": handoff_entries,
            "handoff_suppressed": message.handoff_suppressed,
            "addressed_to_me": participant.id in message.handoff_to_participant_ids,
            "reply_to_message_id": (
                str(message.reply_to_message_id)
                if message.reply_to_message_id
                else None
            ),
            "created_at": message.created_at.isoformat(),
        }

    @staticmethod
    def _turn_status(room: GroupChatRoom) -> dict[str, Any]:
        return {
            "consecutive_agent_turns": room.consecutive_agent_turns,
            "max_consecutive_agent_turns": room.max_consecutive_agent_turns,
            "turn_cap_reached": room.consecutive_agent_turns
            >= room.max_consecutive_agent_turns,
        }

    async def read_messages(
        self, participant: GroupChatParticipant, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        after_seq = arguments.get("after_seq")
        before_seq = arguments.get("before_seq")
        limit = arguments.get("limit", 50)
        if after_seq is not None and (
            isinstance(after_seq, bool)
            or not isinstance(after_seq, int)
            or after_seq < 0
        ):
            raise ServiceError(
                "after_seq must be a non-negative integer.", 422, "invalid_arguments"
            )
        if before_seq is not None and (
            isinstance(before_seq, bool)
            or not isinstance(before_seq, int)
            or before_seq < 1
        ):
            raise ServiceError(
                "before_seq must be a positive integer.", 422, "invalid_arguments"
            )
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not (1 <= limit <= 100)
        ):
            raise ServiceError(
                "limit must be an integer between 1 and 100.", 422, "invalid_arguments"
            )

        page = await self.services.group_chat.list_messages(
            participant.room_id,
            after_seq=after_seq,
            before_seq=before_seq,
            limit=limit,
        )
        roster = await self.services.group_chat.list_participants(participant.room_id)
        by_id = {item.id: item for item in roster}
        formatted = [
            self.format_message(msg, participant=participant, participants_by_id=by_id)
            for msg in page.messages
        ]
        if page.messages:
            max_seq = max(msg.seq for msg in page.messages)
            try:
                await self.services.group_chat.touch_participant(
                    participant.room_id,
                    participant.id,
                    last_seen_at=datetime.now(UTC),
                    cursor_seq=max_seq,
                )
            except ServiceError:
                pass
        return {
            "room_id": str(page.room.id),
            "room_title": page.room.title,
            "messages": formatted,
            "latest_seq": page.latest_seq,
            "has_more": page.has_more,
            **self._turn_status(page.room),
        }

    def _resolve_handoff_targets(
        self,
        raw_handoff: Any,
        *,
        sender: GroupChatParticipant,
        roster: list[GroupChatParticipant],
    ) -> list[GroupChatParticipant]:
        if raw_handoff is None:
            return []
        if isinstance(raw_handoff, str):
            raw_list = [raw_handoff] if raw_handoff.strip() else []
        elif isinstance(raw_handoff, list):
            raw_list = [
                item for item in raw_handoff if isinstance(item, str) and item.strip()
            ]
        else:
            raise ServiceError(
                "handoff_to must be a participant handle/ID or list of handles/IDs.",
                422,
                "invalid_handoff",
            )
        if len(raw_list) > 10:
            raise ServiceError(
                "At most 10 handoff targets are allowed.", 422, "invalid_handoff"
            )

        by_id = {str(p.id).lower(): p for p in roster}
        by_handle = {p.handle.lower(): p for p in roster}
        resolved: list[GroupChatParticipant] = []
        seen: set[UUID] = set()
        for raw_ref in raw_list:
            cleaned = raw_ref.strip().lstrip("@").lower()
            target = by_id.get(cleaned) or by_handle.get(cleaned)
            if target is None:
                raise ServiceError(
                    f"Participant '{raw_ref}' was not found in Group Chat.",
                    422,
                    "unknown_participant",
                )
            if target.id == sender.id:
                raise ServiceError(
                    "An agent cannot hand off a turn to itself.",
                    422,
                    "self_handoff_not_allowed",
                )
            if target.revoked_at is not None or not target.enabled:
                raise ServiceError(
                    f"Participant '@{target.handle}' is not active.",
                    409,
                    "participant_inactive",
                )
            if target.id not in seen:
                seen.add(target.id)
                resolved.append(target)
        return resolved

    async def send_message(
        self, participant: GroupChatParticipant, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        content = arguments.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ServiceError(
                "Message content cannot be blank.", 422, "invalid_arguments"
            )
        if len(content) > 16000:
            raise ServiceError(
                "Message content exceeds 16,000 characters.", 422, "invalid_arguments"
            )

        reply_to_raw = arguments.get("reply_to_message_id")
        reply_to_id: UUID | None = None
        if reply_to_raw is not None:
            if not isinstance(reply_to_raw, str):
                raise ServiceError(
                    "reply_to_message_id must be a UUID string.",
                    422,
                    "invalid_arguments",
                )
            try:
                reply_to_id = UUID(reply_to_raw)
            except ValueError as error:
                raise ServiceError(
                    "reply_to_message_id must be a valid UUID.",
                    422,
                    "invalid_arguments",
                ) from error

        client_request_id = arguments.get("client_request_id")
        if client_request_id is not None and not isinstance(client_request_id, str):
            raise ServiceError(
                "client_request_id must be a string.", 422, "invalid_arguments"
            )

        roster = await self.services.group_chat.list_participants(participant.room_id)
        targets = self._resolve_handoff_targets(
            arguments.get("handoff_to"), sender=participant, roster=roster
        )
        result = await self.services.group_chat.create_message(
            participant.room_id,
            sender_kind=participant.kind,
            participant_id=participant.id,
            sender_name=participant.display_name,
            content=content,
            handoff_to_participant_ids=[item.id for item in targets],
            reply_to_message_id=reply_to_id,
            client_request_id=client_request_id,
        )
        try:
            await self.services.group_chat.touch_participant(
                participant.room_id,
                participant.id,
                last_seen_at=datetime.now(UTC),
                cursor_seq=result.message.seq,
            )
        except ServiceError:
            pass

        by_id = {item.id: item for item in roster}
        effective_handoff = [
            {
                "id": str(target_id),
                "handle": by_id[target_id].handle
                if target_id in by_id
                else str(target_id),
                "display_name": (
                    by_id[target_id].display_name
                    if target_id in by_id
                    else str(target_id)
                ),
            }
            for target_id in result.effective_handoff_to_participant_ids
        ]
        return {
            "id": str(result.message.id),
            "seq": result.message.seq,
            "room_id": str(result.room.id),
            "sender_kind": result.message.sender_kind,
            "sender_name": result.message.sender_name,
            "handoff_to": effective_handoff,
            "handoff_suppressed": result.handoff_suppressed,
            "idempotent_replay": result.idempotent_replay,
            **self._turn_status(result.room),
        }

    async def wait_for_messages(
        self, participant: GroupChatParticipant, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        after_seq = arguments.get("after_seq")
        if after_seq is None:
            after_seq = participant.cursor_seq
        if (
            isinstance(after_seq, bool)
            or not isinstance(after_seq, int)
            or after_seq < 0
        ):
            raise ServiceError(
                "after_seq must be a non-negative integer.", 422, "invalid_arguments"
            )

        timeout_seconds = arguments.get("timeout_seconds", DEFAULT_WAIT_TIMEOUT_SECONDS)
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ServiceError(
                "timeout_seconds must be a positive number.", 422, "invalid_arguments"
            )
        timeout = min(float(timeout_seconds), float(MAX_WAIT_TIMEOUT_SECONDS))

        only_when_addressed = arguments.get("only_when_addressed", True)
        if not isinstance(only_when_addressed, bool):
            raise ServiceError(
                "only_when_addressed must be a boolean.", 422, "invalid_arguments"
            )

        limit = arguments.get("limit", 50)
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not (1 <= limit <= 100)
        ):
            raise ServiceError(
                "limit must be an integer between 1 and 100.", 422, "invalid_arguments"
            )

        async with self.rate_limiter.wait_slot(participant.id):
            deadline = time.monotonic() + timeout
            while True:
                page = await self.services.group_chat.list_messages(
                    participant.room_id, after_seq=after_seq, limit=limit
                )
                at_cap = (
                    page.room.consecutive_agent_turns
                    >= page.room.max_consecutive_agent_turns
                )
                matched = False
                trigger_seq: int | None = None
                if page.messages:
                    if only_when_addressed:
                        if not at_cap:
                            addressed = [
                                msg
                                for msg in page.messages
                                if participant.id in msg.handoff_to_participant_ids
                                and not msg.handoff_suppressed
                            ]
                            if addressed:
                                matched = True
                                trigger_seq = addressed[-1].seq
                    else:
                        matched = True
                        trigger_seq = page.messages[-1].seq

                if matched:
                    roster = await self.services.group_chat.list_participants(
                        participant.room_id
                    )
                    by_id = {item.id: item for item in roster}
                    formatted = [
                        self.format_message(
                            msg, participant=participant, participants_by_id=by_id
                        )
                        for msg in page.messages
                    ]
                    max_seq = max(msg.seq for msg in page.messages)
                    try:
                        await self.services.group_chat.touch_participant(
                            participant.room_id,
                            participant.id,
                            last_seen_at=datetime.now(UTC),
                            cursor_seq=max_seq,
                        )
                    except ServiceError:
                        pass
                    return {
                        "room_id": str(page.room.id),
                        "room_title": page.room.title,
                        "messages": formatted,
                        "latest_seq": page.latest_seq,
                        "trigger_seq": trigger_seq,
                        "has_more": page.has_more,
                        "timed_out": False,
                        **self._turn_status(page.room),
                    }

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {
                        "room_id": str(page.room.id),
                        "room_title": page.room.title,
                        "messages": [],
                        "latest_seq": page.latest_seq,
                        "trigger_seq": None,
                        "has_more": False,
                        "timed_out": True,
                        **self._turn_status(page.room),
                    }
                await asyncio.sleep(min(self.wait_poll_interval, remaining))

    async def subscribe_events(
        self, participant: GroupChatParticipant, params: dict[str, Any]
    ) -> dict[str, Any]:
        if participant.kind != "external_agent":
            raise ServiceError(
                "Only external agents can subscribe to webhook events.",
                403,
                "external_only",
            )
        event_name = (
            params.get("event") or params.get("eventName") or params.get("name")
        )
        if event_name != EVENT_TURN_REQUESTED:
            raise ServiceError(
                f"Unsupported event '{event_name}'. Supported events: {EVENT_TURN_REQUESTED}.",
                422,
                "unsupported_event",
            )

        webhook_cfg = params.get("webhook") or params.get("delivery") or {}
        if webhook_cfg is not None and not isinstance(webhook_cfg, dict):
            raise ServiceError(
                "webhook configuration must be an object.", 422, "invalid_arguments"
            )
        callback_url_raw = (
            webhook_cfg.get("url")
            or webhook_cfg.get("callbackUrl")
            or params.get("callbackUrl")
            or params.get("webhookUrl")
            or params.get("callback_url")
        )
        signing_secret_raw = (
            webhook_cfg.get("secret")
            or webhook_cfg.get("signingSecret")
            or params.get("signingSecret")
            or params.get("signing_secret")
            or params.get("secret")
        )
        if not isinstance(callback_url_raw, str) or not callback_url_raw.strip():
            raise ServiceError(
                "A webhook callback URL is required.", 422, "invalid_arguments"
            )
        if not isinstance(signing_secret_raw, str) or not signing_secret_raw.strip():
            raise ServiceError(
                "A Standard Webhooks signing secret (whsec_...) is required.",
                422,
                "invalid_arguments",
            )
        try:
            callback_url = validate_webhook_callback_url(callback_url_raw)
            decode_whsec_secret(signing_secret_raw)
        except ValueError as error:
            raise ServiceError(str(error), 422, "invalid_arguments") from error

        raw_args = (
            params.get("arguments") if "arguments" in params else params.get("filters")
        )
        filters = raw_args if raw_args is not None else {}
        if not isinstance(filters, dict):
            raise ServiceError("arguments must be an object.", 422, "invalid_arguments")
        if "room_id" in filters and str(filters["room_id"]) != str(participant.room_id):
            raise ServiceError(
                "room_id argument does not match this participant's room.",
                422,
                "invalid_filter",
            )

        ttl_raw = (
            params.get("ttlSeconds")
            or params.get("ttl")
            or params.get("ttl_seconds")
            or DEFAULT_SUBSCRIPTION_TTL_SECONDS
        )
        if isinstance(ttl_raw, bool) or not isinstance(ttl_raw, (int, float)):
            raise ServiceError("ttlSeconds must be a number.", 422, "invalid_arguments")
        ttl_seconds = max(
            MIN_SUBSCRIPTION_TTL_SECONDS,
            min(MAX_SUBSCRIPTION_TTL_SECONDS, int(ttl_raw)),
        )

        # Reuse one durable subscription ID per (room, participant, event) identity.
        existing_subs = await self.services.group_chat.list_subscriptions(
            participant.room_id, participant_id=participant.id
        )
        existing_match = next(
            (sub for sub in existing_subs if sub.event_name == event_name),
            None,
        )
        subscription_id = (
            existing_match.id
            if existing_match is not None
            else uuid5(
                participant.room_id, f"group-chat-sub:{participant.id}:{event_name}"
            )
        )
        await self._verify_webhook_challenge(
            subscription_id=subscription_id,
            callback_url=callback_url,
            signing_secret=signing_secret_raw,
            event_name=event_name,
        )

        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=ttl_seconds)
        refresh_margin = min(300, max(1, ttl_seconds // 10))
        refresh_before = expires_at - timedelta(seconds=refresh_margin)
        saved = await self.services.group_chat.create_subscription(
            participant.room_id,
            subscription_id=subscription_id,
            participant_id=participant.id,
            event_name=event_name,
            filters=filters,
            callback_url=callback_url,
            signing_secret=signing_secret_raw,
            expires_at=expires_at,
        )
        return {
            "id": str(saved.id),
            "event": saved.event_name,
            "status": saved.status,
            "refreshBefore": refresh_before.isoformat(),
        }

    async def _verify_webhook_challenge(
        self,
        *,
        subscription_id: UUID,
        callback_url: str,
        signing_secret: str,
        event_name: str,
    ) -> None:
        challenge = secrets.token_urlsafe(24)
        event_id = str(uuid4())
        timestamp = int(time.time())
        body_bytes = json.dumps(
            {
                "type": "mcp.events.challenge",
                "challenge": challenge,
                "subscriptionId": str(subscription_id),
                "event": event_name,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        headers = build_webhook_headers(
            subscription_id=str(subscription_id),
            event_id=event_id,
            timestamp=timestamp,
            signing_secret=signing_secret,
            payload_bytes=body_bytes,
        )
        try:
            response = await self.services.client.post(
                callback_url,
                content=body_bytes,
                headers=headers,
                follow_redirects=False,
                timeout=httpx.Timeout(CHALLENGE_TIMEOUT_SECONDS, connect=5),
            )
        except httpx.RequestError as error:
            raise ServiceError(
                f"Webhook challenge verification could not reach {callback_url}: {error}",
                422,
                "challenge_failed",
            ) from error
        if 300 <= response.status_code < 400:
            raise ServiceError(
                "Webhook callback URL returned a redirect, which is not permitted.",
                422,
                "challenge_redirect_rejected",
            )
        if not response.is_success:
            raise ServiceError(
                f"Webhook challenge verification failed with HTTP {response.status_code}.",
                422,
                "challenge_failed",
            )
        echoed: str | None = None
        try:
            parsed = response.json()
            if isinstance(parsed, dict) and isinstance(parsed.get("challenge"), str):
                echoed = parsed["challenge"]
        except ValueError:
            echoed = response.text.strip()
        if echoed != challenge:
            raise ServiceError(
                "Webhook callback did not return the expected verification challenge.",
                422,
                "challenge_mismatch",
            )

    async def unsubscribe_events(
        self, participant: GroupChatParticipant, params: dict[str, Any]
    ) -> dict[str, Any]:
        sub_raw = (
            params.get("id")
            or params.get("subscriptionId")
            or params.get("subscription_id")
        )
        if not isinstance(sub_raw, str) or not sub_raw.strip():
            raise ServiceError("id is required.", 422, "invalid_arguments")
        try:
            subscription_id = UUID(sub_raw.strip())
        except ValueError as error:
            raise ServiceError(
                "id must be a valid UUID.", 422, "invalid_arguments"
            ) from error
        await self.services.group_chat.revoke_subscription(
            subscription_id, participant_id=participant.id
        )
        return {
            "id": str(subscription_id),
            "unsubscribed": True,
        }
