"""Kadan client for Tsuchi Group Chat storage, token hashing, and directed Kadan agent wake-ups."""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid5

from .config import ServiceError
from .pydantic_models.agent_runs import AgentRun
from .pydantic_models.group_chat import (
    GroupChatDeliveryStatus,
    GroupChatEventSubscription,
    GroupChatMessagesPage,
    GroupChatMessageWriteResult,
    GroupChatParticipant,
    GroupChatParticipantPublic,
    GroupChatPushStatus,
    GroupChatRoom,
    GroupChatSenderKind,
    GroupChatSubscriptionStatus,
    GroupChatWebhookDelivery,
)
from .pydantic_models.pets import PetRef
from .pydantic_models.resources import ResourceKind

if TYPE_CHECKING:
    from .services import LocalServices

logger = logging.getLogger(__name__)

TOKEN_PREFIX = "kdn_gc_"
DEFAULT_AGENT_DESCRIPTION = "Built-in Kadan agent"


def generate_group_chat_token() -> str:
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


def hash_group_chat_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def summarize_participant_status(
    participant: GroupChatParticipant,
    subscriptions: list[GroupChatEventSubscription],
    *,
    now: datetime | None = None,
) -> GroupChatParticipantPublic:
    """Project a participant and its subscriptions into a secret-free browser view."""
    current = now or datetime.now(UTC)
    if participant.kind == "kadan_agent":
        return GroupChatParticipantPublic(
            **participant.model_dump(),
            push_status="built_in",
            push_status_detail="Wakes automatically when tagged or handed the turn.",
            active_subscriptions=0,
        )

    if participant.revoked_at is not None:
        return GroupChatParticipantPublic(
            **participant.model_dump(),
            push_status="revoked",
            push_status_detail="Access revoked. This agent can no longer read or post.",
            active_subscriptions=0,
        )

    if not participant.enabled:
        return GroupChatParticipantPublic(
            **participant.model_dump(),
            push_status="paused",
            push_status_detail="Paused. Resume to let this agent read and reply again.",
            active_subscriptions=0,
        )

    active_live = [
        sub
        for sub in subscriptions
        if sub.status == "active" and sub.expires_at > current
    ]
    if active_live:
        last_error = next(
            (sub.last_error for sub in active_live if sub.last_error), None
        )
        if last_error:
            return GroupChatParticipantPublic(
                **participant.model_dump(),
                push_status="error",
                push_status_detail=f"Kadan could not reach this agent the last time it was tagged: {last_error}",
                active_subscriptions=len(active_live),
            )
        return GroupChatParticipantPublic(
            **participant.model_dump(),
            push_status="active",
            push_status_detail="Kadan lets this agent know whenever it is tagged.",
            active_subscriptions=len(active_live),
        )

    failed = [sub for sub in subscriptions if sub.status == "failed"]
    if failed:
        detail = failed[0].last_error or "Webhook delivery failed."
        return GroupChatParticipantPublic(
            **participant.model_dump(),
            push_status="error",
            push_status_detail=f"Kadan stopped letting this agent know when it is tagged after an error: {detail}",
            active_subscriptions=0,
        )

    expired = [
        sub
        for sub in subscriptions
        if sub.status == "expired"
        or (sub.status == "active" and sub.expires_at <= current)
    ]
    if expired:
        return GroupChatParticipantPublic(
            **participant.model_dump(),
            push_status="expired",
            push_status_detail="This agent stopped listening for tags. It will hear about them again once it reconnects.",
            active_subscriptions=0,
        )

    push_status: GroupChatPushStatus = "idle"
    detail = (
        "Connected. This agent checks Group Chat on its own schedule."
        if participant.last_seen_at is not None
        else "Waiting for this agent to connect with its connection key."
    )
    return GroupChatParticipantPublic(
        **participant.model_dump(),
        push_status=push_status,
        push_status_detail=detail,
        active_subscriptions=0,
    )


class GroupChatClient:
    """Tsuchi storage client and Kadan wake-up coordinator for Group Chat."""

    def __init__(self, services: LocalServices):
        self.services = services

    async def default_room(self) -> GroupChatRoom:
        data = await self.services.json_request(
            "Tsuchi", "GET", "/api/group-chat/rooms/default"
        )
        return GroupChatRoom.model_validate(data)

    async def get_room(self, room_id: UUID) -> GroupChatRoom:
        data = await self.services.json_request(
            "Tsuchi", "GET", f"/api/group-chat/rooms/{room_id}"
        )
        return GroupChatRoom.model_validate(data)

    async def update_room(
        self,
        room_id: UUID,
        *,
        expected_version: int,
        title: str | None = None,
        max_consecutive_agent_turns: int | None = None,
    ) -> GroupChatRoom:
        body: dict[str, Any] = {"expected_version": expected_version}
        if title is not None:
            body["title"] = title
        if max_consecutive_agent_turns is not None:
            body["max_consecutive_agent_turns"] = max_consecutive_agent_turns
        data = await self.services.json_request(
            "Tsuchi", "PUT", f"/api/group-chat/rooms/{room_id}", json=body
        )
        return GroupChatRoom.model_validate(data)

    async def list_participants(self, room_id: UUID) -> list[GroupChatParticipant]:
        rows = await self.services.json_request(
            "Tsuchi", "GET", f"/api/group-chat/rooms/{room_id}/participants"
        )
        return [GroupChatParticipant.model_validate(row) for row in rows]

    async def ensure_kadan_participant(
        self, room_id: UUID, *, agent_key: str | None = None
    ) -> GroupChatParticipant | None:
        """Ensure Kadan's built-in agent is registered as a room participant and synced."""
        resolved_key = agent_key
        resolved_name: str | None = None
        if resolved_key is None:
            agents = await self.services.list_agents()
            if not agents:
                return None
            first = agents[0]
            if isinstance(first, dict):
                resolved_key = str(first.get("key") or "")
                resolved_name = str(first.get("name") or resolved_key)
            else:
                resolved_key = getattr(first, "key", "")
                resolved_name = getattr(first, "name", resolved_key)
        else:
            agent = await self.services.agent(resolved_key)
            resolved_name = agent.name

        if not resolved_key:
            return None
        display_name = (resolved_name or resolved_key).strip() or resolved_key
        data = await self.services.json_request(
            "Tsuchi",
            "POST",
            f"/api/group-chat/rooms/{room_id}/participants",
            json={
                "display_name": display_name,
                "kind": "kadan_agent",
                "agent_key": resolved_key,
                "description": DEFAULT_AGENT_DESCRIPTION,
            },
        )
        return GroupChatParticipant.model_validate(data)

    async def list_public_participants(
        self, room_id: UUID
    ) -> list[GroupChatParticipantPublic]:
        participants = await self.list_participants(room_id)
        subscriptions = await self.list_subscriptions(room_id)
        subs_by_participant: dict[UUID, list[GroupChatEventSubscription]] = {}
        for sub in subscriptions:
            subs_by_participant.setdefault(sub.participant_id, []).append(sub)
        now = datetime.now(UTC)
        return [
            summarize_participant_status(
                participant,
                subs_by_participant.get(participant.id, []),
                now=now,
            )
            for participant in participants
        ]

    async def create_external_participant(
        self,
        room_id: UUID,
        *,
        display_name: str,
        description: str = "",
        handle: str | None = None,
    ) -> tuple[GroupChatParticipant, str]:
        raw_token = generate_group_chat_token()
        payload: dict[str, Any] = {
            "display_name": display_name,
            "kind": "external_agent",
            "description": description,
            "token_hash": hash_group_chat_token(raw_token),
        }
        if handle is not None:
            payload["handle"] = handle
        data = await self.services.json_request(
            "Tsuchi",
            "POST",
            f"/api/group-chat/rooms/{room_id}/participants",
            json=payload,
        )
        return GroupChatParticipant.model_validate(data), raw_token

    async def update_participant(
        self,
        room_id: UUID,
        participant_id: UUID,
        *,
        expected_version: int,
        display_name: str | None = None,
        description: str | None = None,
        enabled: bool | None = None,
        rotate_token: bool = False,
        revoke: bool = False,
        pet: PetRef | None = None,
        clear_pet: bool = False,
    ) -> tuple[GroupChatParticipant, str | None]:
        raw_token: str | None = None
        payload: dict[str, Any] = {
            "expected_version": expected_version,
            "revoke": revoke,
            "clear_pet": clear_pet,
        }
        if pet is not None:
            payload["pet"] = pet.model_dump(mode="json")
        if display_name is not None:
            payload["display_name"] = display_name
        if description is not None:
            payload["description"] = description
        if enabled is not None:
            payload["enabled"] = enabled
        if rotate_token and not revoke:
            raw_token = generate_group_chat_token()
            payload["token_hash"] = hash_group_chat_token(raw_token)
        data = await self.services.json_request(
            "Tsuchi",
            "PUT",
            f"/api/group-chat/rooms/{room_id}/participants/{participant_id}",
            json=payload,
        )
        return GroupChatParticipant.model_validate(data), raw_token

    async def authenticate_token(self, raw_token: str) -> GroupChatParticipant:
        data = await self.services.json_request(
            "Tsuchi",
            "POST",
            "/api/group-chat/participants/authenticate",
            json={"token_hash": hash_group_chat_token(raw_token)},
        )
        return GroupChatParticipant.model_validate(data)

    async def touch_participant(
        self,
        room_id: UUID,
        participant_id: UUID,
        *,
        last_seen_at: datetime | None = None,
        cursor_seq: int | None = None,
    ) -> GroupChatParticipant:
        payload: dict[str, Any] = {}
        if last_seen_at is not None:
            payload["last_seen_at"] = last_seen_at.astimezone(UTC).isoformat()
        if cursor_seq is not None:
            payload["cursor_seq"] = cursor_seq
        data = await self.services.json_request(
            "Tsuchi",
            "POST",
            f"/api/group-chat/rooms/{room_id}/participants/{participant_id}/touch",
            json=payload,
        )
        return GroupChatParticipant.model_validate(data)

    async def list_messages(
        self,
        room_id: UUID,
        *,
        after_seq: int | None = None,
        before_seq: int | None = None,
        limit: int = 50,
    ) -> GroupChatMessagesPage:
        params: dict[str, Any] = {"limit": limit}
        if after_seq is not None:
            params["after_seq"] = after_seq
        if before_seq is not None:
            params["before_seq"] = before_seq
        data = await self.services.json_request(
            "Tsuchi",
            "GET",
            f"/api/group-chat/rooms/{room_id}/messages",
            params=params,
        )
        return GroupChatMessagesPage.model_validate(data)

    async def create_message(
        self,
        room_id: UUID,
        *,
        sender_kind: GroupChatSenderKind,
        sender_name: str,
        content: str,
        participant_id: UUID | None = None,
        handoff_to_participant_ids: list[UUID] | None = None,
        reply_to_message_id: UUID | None = None,
        client_request_id: str | None = None,
        trigger_kadan_wake: bool = True,
    ) -> GroupChatMessageWriteResult:
        payload: dict[str, Any] = {
            "sender_kind": sender_kind,
            "participant_id": str(participant_id) if participant_id else None,
            "sender_name": sender_name,
            "content": content,
            "handoff_to_participant_ids": [
                str(pid) for pid in (handoff_to_participant_ids or [])
            ],
            "reply_to_message_id": (
                str(reply_to_message_id) if reply_to_message_id else None
            ),
            "client_request_id": client_request_id,
        }
        data = await self.services.json_request(
            "Tsuchi",
            "POST",
            f"/api/group-chat/rooms/{room_id}/messages",
            json=payload,
        )
        result = GroupChatMessageWriteResult.model_validate(data)
        if (
            trigger_kadan_wake
            and not result.idempotent_replay
            and not result.turn_cap_reached
            and result.effective_handoff_to_participant_ids
        ):
            try:
                await self.maybe_wake_kadan_agents(
                    room_id,
                    result.effective_handoff_to_participant_ids,
                    trigger_seq=result.message.seq,
                    sender_participant_id=participant_id,
                )
            except ServiceError as error:
                logger.warning(
                    "Could not immediately queue Kadan Group Chat wake-up for seq=%s: %s",
                    result.message.seq,
                    error,
                )
        return result

    async def maybe_wake_kadan_agents(
        self,
        room_id: UUID,
        target_participant_ids: list[UUID],
        *,
        trigger_seq: int,
        sender_participant_id: UUID | None = None,
        participants: list[GroupChatParticipant] | None = None,
        skip_if_running: bool = False,
    ) -> list[AgentRun]:
        """Queue coalesced idempotent group_chat event runs for targeted Kadan agents."""
        if not target_participant_ids:
            return []
        roster = (
            participants
            if participants is not None
            else await self.list_participants(room_id)
        )
        by_id = {item.id: item for item in roster}
        queued: list[AgentRun] = []
        for target_id in dict.fromkeys(target_participant_ids):
            if sender_participant_id is not None and target_id == sender_participant_id:
                continue
            target = by_id.get(target_id)
            if (
                target is None
                or target.kind != "kadan_agent"
                or not target.agent_key
                or not target.enabled
                or target.revoked_at is not None
            ):
                continue
            conversation_id = await self._ensure_agent_conversation_id(target.agent_key)
            if conversation_id is None:
                continue
            if await self._has_coalescable_group_chat_run(
                target.agent_key,
                conversation_id,
                room_id,
                include_running=skip_if_running,
            ):
                continue
            event_id = uuid5(
                room_id,
                f"group-chat-wake:{target.id}:after-{target.cursor_seq}:{trigger_seq}",
            )
            payload = {
                "room_id": str(room_id),
                "participant_id": str(target.id),
                "trigger_seq": trigger_seq,
            }
            created = await self.services.json_request(
                "Tsuchi",
                "POST",
                f"/api/agents/{target.agent_key}/resources/{conversation_id}/event-runs",
                json={
                    "id": str(event_id),
                    "event": {
                        "kind": "group_chat",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                },
            )
            queued.append(AgentRun.model_validate(created))
        return queued

    async def _ensure_agent_conversation_id(self, agent_key: str) -> UUID | None:
        conversations = await self.services.list_resource_kinds(
            agent_key, (ResourceKind.CONVERSATION,), limit=10
        )
        if conversations:
            conversations.sort(
                key=lambda item: (item.updated_at, item.created_at), reverse=True
            )
            return conversations[0].id
        created = await self.services.conversations.create(agent_key)
        return created.id

    async def _has_coalescable_group_chat_run(
        self,
        agent_key: str,
        conversation_id: UUID,
        room_id: UUID,
        *,
        include_running: bool = False,
    ) -> bool:
        try:
            values = await self.services.json_request(
                "Tsuchi",
                "GET",
                f"/api/agents/{agent_key}/resources/{conversation_id}/runs",
            )
        except ServiceError:
            return False
        if not isinstance(values, list):
            return False
        statuses = {"queued", "running"} if include_running else {"queued"}
        for value in values:
            try:
                run = AgentRun.model_validate(value)
            except Exception:
                continue
            if run.status not in statuses:
                continue
            if run.trigger_event is None or run.trigger_event.kind != "group_chat":
                continue
            try:
                detail = json.loads(run.trigger_event.content)
            except ValueError:
                detail = {}
            if isinstance(detail, dict) and detail.get("room_id") == str(room_id):
                return True
        return False

    async def list_subscriptions(
        self,
        room_id: UUID,
        *,
        participant_id: UUID | None = None,
        status: GroupChatSubscriptionStatus | None = None,
    ) -> list[GroupChatEventSubscription]:
        params: dict[str, Any] = {}
        if participant_id is not None:
            params["participant_id"] = str(participant_id)
        if status is not None:
            params["status"] = status
        rows = await self.services.json_request(
            "Tsuchi",
            "GET",
            f"/api/group-chat/rooms/{room_id}/subscriptions",
            params=params,
        )
        return [GroupChatEventSubscription.model_validate(row) for row in rows]

    async def get_subscription(
        self, subscription_id: UUID
    ) -> GroupChatEventSubscription:
        data = await self.services.json_request(
            "Tsuchi", "GET", f"/api/group-chat/subscriptions/{subscription_id}"
        )
        return GroupChatEventSubscription.model_validate(data)

    async def create_subscription(
        self,
        room_id: UUID,
        *,
        subscription_id: UUID,
        participant_id: UUID,
        event_name: str,
        filters: dict[str, Any],
        callback_url: str,
        signing_secret: str,
        expires_at: datetime,
    ) -> GroupChatEventSubscription:
        data = await self.services.json_request(
            "Tsuchi",
            "POST",
            f"/api/group-chat/rooms/{room_id}/subscriptions",
            json={
                "id": str(subscription_id),
                "participant_id": str(participant_id),
                "event_name": event_name,
                "filters": filters,
                "callback_url": callback_url,
                "signing_secret": signing_secret,
                "expires_at": expires_at.astimezone(UTC).isoformat(),
            },
        )
        return GroupChatEventSubscription.model_validate(data)

    async def update_subscription(
        self,
        subscription_id: UUID,
        *,
        expected_version: int,
        status: GroupChatSubscriptionStatus | None = None,
        expires_at: datetime | None = None,
        last_delivered_seq: int | None = None,
        last_error: str | None = None,
        set_last_error: bool = False,
    ) -> GroupChatEventSubscription:
        payload: dict[str, Any] = {"expected_version": expected_version}
        if status is not None:
            payload["status"] = status
        if expires_at is not None:
            payload["expires_at"] = expires_at.astimezone(UTC).isoformat()
        if last_delivered_seq is not None:
            payload["last_delivered_seq"] = last_delivered_seq
        if set_last_error or last_error is not None:
            payload["last_error"] = last_error
        data = await self.services.json_request(
            "Tsuchi",
            "PUT",
            f"/api/group-chat/subscriptions/{subscription_id}",
            json=payload,
        )
        return GroupChatEventSubscription.model_validate(data)

    async def revoke_subscription(
        self, subscription_id: UUID, *, participant_id: UUID | None = None
    ) -> None:
        params = (
            {"participant_id": str(participant_id)}
            if participant_id is not None
            else None
        )
        await self.services.json_request(
            "Tsuchi",
            "DELETE",
            f"/api/group-chat/subscriptions/{subscription_id}",
            params=params,
        )

    async def list_webhook_deliveries(
        self,
        *,
        status: GroupChatDeliveryStatus | None = None,
        due_before: datetime | None = None,
        room_id: UUID | None = None,
        participant_id: UUID | None = None,
        limit: int = 100,
    ) -> list[GroupChatWebhookDelivery]:
        params: dict[str, Any] = {"limit": limit}
        if status is not None:
            params["status"] = status
        if due_before is not None:
            params["due_before"] = due_before.astimezone(UTC).isoformat()
        if room_id is not None:
            params["room_id"] = str(room_id)
        if participant_id is not None:
            params["participant_id"] = str(participant_id)
        rows = await self.services.json_request(
            "Tsuchi",
            "GET",
            "/api/group-chat/webhook-deliveries",
            params=params,
        )
        return [GroupChatWebhookDelivery.model_validate(row) for row in rows]

    async def update_webhook_delivery(
        self,
        delivery_id: UUID,
        *,
        status: GroupChatDeliveryStatus,
        attempts: int,
        expected_version: int,
        next_attempt_at: datetime | None = None,
        last_error: str | None = None,
    ) -> GroupChatWebhookDelivery:
        payload: dict[str, Any] = {
            "status": status,
            "attempts": attempts,
            "expected_version": expected_version,
            "last_error": last_error,
        }
        if next_attempt_at is not None:
            payload["next_attempt_at"] = next_attempt_at.astimezone(UTC).isoformat()
        data = await self.services.json_request(
            "Tsuchi",
            "PUT",
            f"/api/group-chat/webhook-deliveries/{delivery_id}",
            json=payload,
        )
        return GroupChatWebhookDelivery.model_validate(data)
