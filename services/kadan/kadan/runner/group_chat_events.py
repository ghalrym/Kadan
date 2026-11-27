"""Outbound MCP Events signed webhook dispatcher and Kadan Group Chat wake-up reconciler."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID

import httpx

from ..api.config import ServiceError
from ..api.group_chat_mcp import EVENT_TURN_REQUESTED
from ..api.group_chat_webhooks import (
    build_webhook_headers,
    encode_bounded_webhook_payload,
)
from ..api.pydantic_models.group_chat import (
    GroupChatEventSubscription,
    GroupChatParticipant,
    GroupChatRoom,
    GroupChatWebhookDelivery,
)

if TYPE_CHECKING:
    from ..api.services import LocalServices

logger = logging.getLogger(__name__)
POLL_SECONDS = 2.0
MAX_DELIVERY_ATTEMPTS = 5
BASE_BACKOFF_SECONDS = 5
MAX_BACKOFF_SECONDS = 300
WEBHOOK_TIMEOUT_SECONDS = 15.0
PERMANENT_HTTP_STATUSES = frozenset({400, 401, 403, 404, 405, 410, 422})
SUBSCRIPTION_FATAL_STATUSES = frozenset({404, 410})


def compute_retry_backoff(attempt: int) -> timedelta:
    seconds = min(
        MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * (2 ** max(0, attempt - 1))
    )
    return timedelta(seconds=seconds)


class GroupChatWebhookEvents:
    """Dispatches durable signed MCP Event webhooks and reconciles missed Kadan wake-ups."""

    def __init__(self, services: LocalServices):
        self.services = services
        self.runner: asyncio.Task | None = None
        self.wakeup = asyncio.Event()

    def start(self) -> None:
        self.runner = asyncio.create_task(self.run(), name="group-chat-webhook-events")

    async def close(self) -> None:
        if self.runner:
            self.runner.cancel()
            await asyncio.gather(self.runner, return_exceptions=True)

    async def run(self) -> None:
        while True:
            try:
                await self.poll_once()
            except ServiceError as error:
                logger.warning("Group Chat event dispatcher poll failed: %s", error)
            except Exception:
                logger.exception("Unexpected error in Group Chat event dispatcher")
            try:
                await asyncio.wait_for(self.wakeup.wait(), timeout=POLL_SECONDS)
            except TimeoutError:
                pass
            finally:
                self.wakeup.clear()

    async def poll_once(self, *, now: datetime | None = None) -> None:
        current = now or datetime.now(UTC)
        await self.dispatch_pending_webhooks(now=current)
        await self.reconcile_kadan_wakeups()

    async def dispatch_pending_webhooks(self, *, now: datetime | None = None) -> None:
        current = now or datetime.now(UTC)
        deliveries = await self.services.group_chat.list_webhook_deliveries(
            status="pending", due_before=current, limit=100
        )
        if not deliveries:
            return

        rooms_cache: dict[UUID, GroupChatRoom] = {}
        participants_cache: dict[UUID, dict[UUID, GroupChatParticipant]] = {}
        subs_cache: dict[UUID, GroupChatEventSubscription | None] = {}

        for delivery in deliveries:
            try:
                await self._dispatch_one(
                    delivery,
                    now=current,
                    rooms_cache=rooms_cache,
                    participants_cache=participants_cache,
                    subs_cache=subs_cache,
                )
            except ServiceError as error:
                if error.status == 409:
                    continue
                logger.warning(
                    "Could not process Group Chat webhook delivery %s: %s",
                    delivery.id,
                    error,
                )

    async def _dispatch_one(
        self,
        delivery: GroupChatWebhookDelivery,
        *,
        now: datetime,
        rooms_cache: dict[UUID, GroupChatRoom],
        participants_cache: dict[UUID, dict[UUID, GroupChatParticipant]],
        subs_cache: dict[UUID, GroupChatEventSubscription | None],
    ) -> None:
        if delivery.subscription_id not in subs_cache:
            try:
                subs_cache[
                    delivery.subscription_id
                ] = await self.services.group_chat.get_subscription(
                    delivery.subscription_id
                )
            except ServiceError as error:
                if error.status == 404:
                    subs_cache[delivery.subscription_id] = None
                else:
                    raise
        sub = subs_cache[delivery.subscription_id]
        if sub is None or sub.status != "active":
            await self.services.group_chat.update_webhook_delivery(
                delivery.id,
                status="dropped",
                attempts=delivery.attempts,
                expected_version=delivery.version,
                last_error="Subscription is no longer active.",
            )
            return

        if sub.expires_at <= now:
            updated_sub = await self.services.group_chat.update_subscription(
                sub.id,
                expected_version=sub.version,
                status="expired",
                last_error="Push wake-up subscription expired.",
                set_last_error=True,
            )
            subs_cache[sub.id] = updated_sub
            await self.services.group_chat.update_webhook_delivery(
                delivery.id,
                status="dropped",
                attempts=delivery.attempts,
                expected_version=delivery.version,
                last_error="Subscription expired before delivery.",
            )
            return

        if delivery.room_id not in rooms_cache:
            rooms_cache[delivery.room_id] = await self.services.group_chat.get_room(
                delivery.room_id
            )
        room = rooms_cache[delivery.room_id]

        if delivery.room_id not in participants_cache:
            roster = await self.services.group_chat.list_participants(delivery.room_id)
            participants_cache[delivery.room_id] = {item.id: item for item in roster}
        participant = participants_cache[delivery.room_id].get(delivery.participant_id)
        if (
            participant is None
            or participant.revoked_at is not None
            or not participant.enabled
        ):
            updated_sub = await self.services.group_chat.update_subscription(
                sub.id,
                expected_version=sub.version,
                status="revoked",
                last_error="Participant was paused or revoked.",
                set_last_error=True,
            )
            subs_cache[sub.id] = updated_sub
            await self.services.group_chat.update_webhook_delivery(
                delivery.id,
                status="dropped",
                attempts=delivery.attempts,
                expected_version=delivery.version,
                last_error="Participant was paused or revoked.",
            )
            return

        if room.consecutive_agent_turns >= room.max_consecutive_agent_turns:
            await self.services.group_chat.update_webhook_delivery(
                delivery.id,
                status="dropped",
                attempts=delivery.attempts,
                expected_version=delivery.version,
                last_error="Suppressed because Group Chat reached the agent turn limit.",
            )
            return

        payload_data: dict[str, Any] = (
            dict(delivery.payload) if isinstance(delivery.payload, dict) else {}
        )
        payload_bytes = encode_bounded_webhook_payload(
            event_id=str(delivery.id),
            name=EVENT_TURN_REQUESTED,
            timestamp_iso=delivery.created_at.astimezone(UTC).isoformat(),
            cursor=str(delivery.seq),
            data=payload_data,
        )
        timestamp = int(now.timestamp())
        headers = build_webhook_headers(
            subscription_id=str(sub.id),
            event_id=str(delivery.id),
            timestamp=timestamp,
            signing_secret=sub.signing_secret,
            payload_bytes=payload_bytes,
        )

        next_attempts = delivery.attempts + 1
        try:
            response = await self.services.client.post(
                sub.callback_url,
                content=payload_bytes,
                headers=headers,
                follow_redirects=False,
                timeout=httpx.Timeout(WEBHOOK_TIMEOUT_SECONDS, connect=5),
            )
        except httpx.RequestError as error:
            await self._record_transient_failure(
                delivery,
                sub,
                attempts=next_attempts,
                now=now,
                error_detail=f"Network error delivering webhook: {error}"[:500],
                subs_cache=subs_cache,
            )
            return

        if 200 <= response.status_code < 300:
            await self.services.group_chat.update_webhook_delivery(
                delivery.id,
                status="delivered",
                attempts=next_attempts,
                expected_version=delivery.version,
                last_error=None,
            )
            # Refresh cached subscription version since Tsuchi bumped it on delivery.
            subs_cache.pop(sub.id, None)
            return

        if 300 <= response.status_code < 400:
            error_detail = f"Webhook endpoint returned redirect HTTP {response.status_code}, which is not allowed."
            await self.services.group_chat.update_webhook_delivery(
                delivery.id,
                status="failed",
                attempts=next_attempts,
                expected_version=delivery.version,
                last_error=error_detail,
            )
            updated_sub = await self.services.group_chat.update_subscription(
                sub.id,
                expected_version=sub.version,
                status="failed",
                last_error=error_detail,
                set_last_error=True,
            )
            subs_cache[sub.id] = updated_sub
            return

        if response.status_code in PERMANENT_HTTP_STATUSES:
            error_detail = (
                f"Webhook endpoint rejected the event (HTTP {response.status_code})."
            )
            await self.services.group_chat.update_webhook_delivery(
                delivery.id,
                status="failed",
                attempts=next_attempts,
                expected_version=delivery.version,
                last_error=error_detail,
            )
            next_sub_status = (
                "failed"
                if response.status_code in SUBSCRIPTION_FATAL_STATUSES
                else None
            )
            updated_sub = await self.services.group_chat.update_subscription(
                sub.id,
                expected_version=sub.version,
                status=next_sub_status,
                last_error=error_detail,
                set_last_error=True,
            )
            subs_cache[sub.id] = updated_sub
            return

        await self._record_transient_failure(
            delivery,
            sub,
            attempts=next_attempts,
            now=now,
            error_detail=f"Webhook endpoint returned HTTP {response.status_code}.",
            subs_cache=subs_cache,
        )

    async def _record_transient_failure(
        self,
        delivery: GroupChatWebhookDelivery,
        sub: GroupChatEventSubscription,
        *,
        attempts: int,
        now: datetime,
        error_detail: str,
        subs_cache: dict[UUID, GroupChatEventSubscription | None],
    ) -> None:
        if attempts < MAX_DELIVERY_ATTEMPTS:
            next_at = now + compute_retry_backoff(attempts)
            await self.services.group_chat.update_webhook_delivery(
                delivery.id,
                status="pending",
                attempts=attempts,
                next_attempt_at=next_at,
                expected_version=delivery.version,
                last_error=error_detail,
            )
            updated_sub = await self.services.group_chat.update_subscription(
                sub.id,
                expected_version=sub.version,
                last_error=error_detail,
                set_last_error=True,
            )
            subs_cache[sub.id] = updated_sub
        else:
            await self.services.group_chat.update_webhook_delivery(
                delivery.id,
                status="failed",
                attempts=attempts,
                expected_version=delivery.version,
                last_error=f"Delivery failed after {attempts} attempts: {error_detail}",
            )
            updated_sub = await self.services.group_chat.update_subscription(
                sub.id,
                expected_version=sub.version,
                last_error=f"Delivery failed after {attempts} attempts: {error_detail}",
                set_last_error=True,
            )
            subs_cache[sub.id] = updated_sub

    async def reconcile_kadan_wakeups(self) -> None:
        """Ensure any unread message addressed to Kadan's agent triggers one coalesced wake run."""
        try:
            room = await self.services.group_chat.default_room()
        except ServiceError:
            return
        if room.consecutive_agent_turns >= room.max_consecutive_agent_turns:
            return
        participants = await self.services.group_chat.list_participants(room.id)
        kadan_participants = [
            p
            for p in participants
            if p.kind == "kadan_agent"
            and p.enabled
            and p.revoked_at is None
            and p.agent_key
        ]
        for participant in kadan_participants:
            scan_after = participant.cursor_seq
            latest_addressed_seq: int | None = None
            while True:
                page = await self.services.group_chat.list_messages(
                    room.id, after_seq=scan_after, limit=100
                )
                for msg in page.messages:
                    if (
                        participant.id in msg.handoff_to_participant_ids
                        and not msg.handoff_suppressed
                        and msg.participant_id != participant.id
                    ):
                        latest_addressed_seq = msg.seq
                if not page.has_more or not page.messages:
                    break
                next_after = page.messages[-1].seq
                if next_after <= scan_after:
                    break
                scan_after = next_after
            if latest_addressed_seq is None:
                continue
            await self.services.group_chat.maybe_wake_kadan_agents(
                room.id,
                [participant.id],
                trigger_seq=latest_addressed_seq,
                participants=participants,
                skip_if_running=True,
            )
