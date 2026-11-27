"""Automatic rolling compaction for persistent agent conversations."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from ..api.config import ServiceError
from ..api.memory_context import build_core_memories_snapshot
from ..api.pydantic_models.agent_runs import AgentRun
from ..api.pydantic_models.resources import Message, Resource, ResourceKind
from ..api.services import parse_resource
from .agent_inference import AgentInference

if TYPE_CHECKING:
    from ..api.services import LocalServices

logger = logging.getLogger(__name__)
_UNSET: object = object()

COMPACTION_BUDGET_RATIO = 0.65
COMPACTION_MAX_UNCOMPACTED_MESSAGES = 60
IN_RUN_RECENT_TAIL_MESSAGES = 8
IDLE_COMPACTION_DELAY = timedelta(minutes=30)
IDLE_COMPACTION_POLL_SECONDS = 60.0
MAX_SUMMARY_CHARACTERS = 16000
COMPACTION_WRITE_ATTEMPTS = 3
DEFAULT_COMPACTION_CONTEXT = 48000
CHARACTERS_PER_TOKEN = 3
COMPACTION_RESERVED_TOKENS = 4096

SUMMARY_SECTION_PREFIX = (
    "\n\nSummary of earlier conversation (full history remains saved and can be "
    "inspected with search_conversation_history or read_conversation_history when "
    "exact wording is needed):\n"
)

COMPACTION_SYSTEM_PROMPT = (
    "You maintain a concise, accurate rolling summary of an ongoing conversation "
    "between a user and their personal AI assistant.\n"
    "Merge any existing summary with the new messages into one updated summary.\n"
    "Preserve:\n"
    "- Important facts, user preferences, names, goals, constraints, and decisions\n"
    "- Ongoing projects, open tasks, commitments, and unresolved questions\n"
    "- Key outcomes, file/document/memory titles mentioned, and context needed to continue naturally\n"
    "Omit greetings, filler, and redundant repetition. Write clear, compact plain "
    "Markdown (aim for under 3,500 characters). Output only the updated summary, "
    "with no preamble or commentary."
)


def _compaction_context_characters(context_limit: int | None) -> int:
    if context_limit is None:
        return DEFAULT_COMPACTION_CONTEXT
    return max(0, (context_limit - COMPACTION_RESERVED_TOKENS) * CHARACTERS_PER_TOKEN)


def split_uncompacted(
    messages: Sequence[Message],
    compacted_summary: str | None,
    compacted_through_message_id: UUID | None,
) -> tuple[str | None, list[Message]]:
    """Return the active rolling summary and eligible messages after its anchor."""
    cleaned = compacted_summary.strip() if compacted_summary else ""
    if cleaned and compacted_through_message_id is not None:
        anchor_index = next(
            (
                index
                for index, message in enumerate(messages)
                if message.id == compacted_through_message_id
            ),
            None,
        )
        if anchor_index is not None:
            return (
                cleaned,
                [
                    message
                    for message in messages[anchor_index + 1 :]
                    if message.role in {"user", "assistant"}
                ],
            )
    return (
        None,
        [message for message in messages if message.role in {"user", "assistant"}],
    )


def format_summary_section(summary: str | None, max_chars: int | None = None) -> str:
    """Format the rolling summary block appended to system instructions."""
    if not summary or not summary.strip():
        return ""
    body = summary.strip()
    section = SUMMARY_SECTION_PREFIX + body
    if max_chars is None or len(section) <= max_chars:
        return section
    available_body = max_chars - len(SUMMARY_SECTION_PREFIX)
    if available_body <= 1:
        return ""
    return SUMMARY_SECTION_PREFIX + body[: available_body - 1].rstrip() + "…"


def should_compact_in_run(
    uncompacted: Sequence[Message],
    *,
    uncompacted_chars: int,
    summary_chars: int,
    available_budget: int,
) -> bool:
    """Decide whether older uncompacted messages should be folded before inference."""
    if len(uncompacted) <= IN_RUN_RECENT_TAIL_MESSAGES:
        return False
    if len(uncompacted) > COMPACTION_MAX_UNCOMPACTED_MESSAGES:
        return True
    if available_budget <= 0:
        return False
    return (uncompacted_chars + summary_chars) > int(
        available_budget * COMPACTION_BUDGET_RATIO
    )


def format_message_for_compaction(message: Message, max_chars: int) -> str:
    stamp = message.created_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    speaker = "User" if message.role == "user" else "Assistant"
    header = f"[{stamp}] {speaker}: "
    limit = max(32, max_chars - len(header))
    content = message.content.strip()
    if len(content) > limit:
        content = content[: limit - 1].rstrip() + "…"
    return header + content


def chunk_messages_for_compaction(
    messages: Sequence[Message], chunk_char_budget: int
) -> list[list[str]]:
    budget = max(1000, chunk_char_budget)
    chunks: list[list[str]] = []
    current: list[str] = []
    used = 0
    for message in messages:
        line = format_message_for_compaction(message, budget)
        added = len(line) + (1 if current else 0)
        if current and used + added > budget:
            chunks.append(current)
            current = [line]
            used = len(line)
        else:
            current.append(line)
            used += added
    if current:
        chunks.append(current)
    return chunks


def build_compaction_messages(
    existing_summary: str | None, formatted_chunk: Sequence[str]
) -> list[BaseMessage]:
    joined = "\n\n".join(formatted_chunk)
    if existing_summary and existing_summary.strip():
        user_content = (
            "Existing rolling summary:\n"
            f"{existing_summary.strip()}\n\n"
            "New messages to fold into the summary:\n"
            f"{joined}"
        )
    else:
        user_content = f"Messages to summarize:\n{joined}"
    return [
        SystemMessage(content=COMPACTION_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]


async def generate_rolling_summary(
    inference: AgentInference,
    *,
    agent_key: str,
    conversation_id: UUID,
    existing_summary: str | None,
    messages: Sequence[Message],
    model_id: str | None,
    context_limit: int | None = None,
    priority: Literal["interactive", "background"] = "background",
) -> str:
    if not messages:
        raise ValueError("Compaction requires at least one message.")
    rolling = existing_summary.strip() if existing_summary else None
    total_budget = _compaction_context_characters(context_limit)
    usable = max(
        2000,
        int(total_budget * COMPACTION_BUDGET_RATIO)
        - len(COMPACTION_SYSTEM_PROMPT)
        - min(len(rolling or ""), MAX_SUMMARY_CHARACTERS)
        - 512,
    )
    chunks = chunk_messages_for_compaction(messages, usable)
    for chunk in chunks:
        prompt = build_compaction_messages(rolling, chunk)
        rolling = await inference.complete_text(
            prompt,
            model_id,
            conversation_key=None,
            priority=priority,
            reuse_prefix=False,
        )
        rolling = rolling.strip()
        if len(rolling) > MAX_SUMMARY_CHARACTERS:
            rolling = rolling[: MAX_SUMMARY_CHARACTERS - 1].rstrip() + "…"
    assert rolling is not None
    return rolling


async def persist_compaction(
    services: LocalServices,
    *,
    agent_key: str,
    conversation_id: UUID,
    expected_version: int,
    compacted_summary: str,
    compacted_through_message_id: UUID,
    compacted_at: datetime | None = None,
    core_memories_snapshot: str | None | object = _UNSET,
    core_memories_refreshed_at: datetime | None = None,
    require_idle: bool = False,
) -> Resource:
    body: dict[str, object] = {
        "expected_version": expected_version,
        "compacted_summary": compacted_summary,
        "compacted_through_message_id": str(compacted_through_message_id),
        "require_idle": require_idle,
    }
    if compacted_at is not None:
        body["compacted_at"] = compacted_at.astimezone(UTC).isoformat()
    if core_memories_snapshot is not _UNSET:
        body["core_memories_snapshot"] = core_memories_snapshot
        if core_memories_snapshot is not None:
            refreshed = core_memories_refreshed_at or datetime.now(UTC)
            body["core_memories_refreshed_at"] = refreshed.astimezone(UTC).isoformat()
    value = await services.json_request(
        "Tsuchi",
        "PUT",
        f"/api/agents/{agent_key}/resources/{conversation_id}/compaction",
        json=body,
    )
    return parse_resource(value)


def _prefix_matches(
    original: Sequence[Message], current: Sequence[Message], anchor_id: UUID
) -> bool:
    orig_index = next(
        (index for index, item in enumerate(original) if item.id == anchor_id), None
    )
    curr_index = next(
        (index for index, item in enumerate(current) if item.id == anchor_id), None
    )
    if orig_index is None or curr_index is None or orig_index != curr_index:
        return False
    return [item.id for item in original[: orig_index + 1]] == [
        item.id for item in current[: curr_index + 1]
    ]


def _already_compacted_through(conversation: Resource, anchor_id: UUID) -> bool:
    if conversation.compacted_through_message_id is None:
        return False
    indices = {item.id: index for index, item in enumerate(conversation.messages)}
    current_index = indices.get(conversation.compacted_through_message_id)
    target_index = indices.get(anchor_id)
    return (
        current_index is not None
        and target_index is not None
        and current_index >= target_index
    )


async def persist_in_run_compaction(
    services: LocalServices,
    *,
    agent_key: str,
    conversation: Resource,
    compacted_summary: str,
    compacted_through_message_id: UUID,
    core_memories_snapshot: str | None | object = _UNSET,
    core_memories_refreshed_at: datetime | None = None,
    verify_owned: Callable[[], Awaitable[object]] | None = None,
) -> Resource:
    """Persist in-run compaction, tolerating trailing user messages appended concurrently."""
    current = conversation
    for attempt in range(COMPACTION_WRITE_ATTEMPTS):
        if verify_owned is not None:
            await verify_owned()
        try:
            now = datetime.now(UTC)
            return await persist_compaction(
                services,
                agent_key=agent_key,
                conversation_id=current.id,
                expected_version=current.version,
                compacted_summary=compacted_summary,
                compacted_through_message_id=compacted_through_message_id,
                compacted_at=now,
                core_memories_snapshot=core_memories_snapshot,
                core_memories_refreshed_at=core_memories_refreshed_at or now,
                require_idle=False,
            )
        except ServiceError as error:
            if error.status != 409 or attempt == COMPACTION_WRITE_ATTEMPTS - 1:
                raise
            refreshed = await services.resource(agent_key, current.id)
            if _already_compacted_through(refreshed, compacted_through_message_id):
                return refreshed
            if not _prefix_matches(
                conversation.messages,
                refreshed.messages,
                compacted_through_message_id,
            ):
                raise
            current = refreshed
    return current


class IdleCompactor:
    """Background loop that auto-compacts persistent threads idle for > 30 minutes."""

    def __init__(
        self,
        services: LocalServices,
        *,
        agent_keys: Callable[[], Sequence[str]] | None = None,
        inference: AgentInference | None = None,
        idle_delay: timedelta = IDLE_COMPACTION_DELAY,
        poll_seconds: float = IDLE_COMPACTION_POLL_SECONDS,
    ):
        self.services = services
        self._agent_keys = agent_keys
        self.inference = inference or AgentInference(
            services.client, services.settings.hikari_url
        )
        self.idle_delay = idle_delay
        self.poll_seconds = poll_seconds
        self.runner: asyncio.Task | None = None

    def set_agent_keys_provider(
        self, provider: Callable[[], Sequence[str]]
    ) -> None:
        self._agent_keys = provider

    def start(self) -> None:
        self.runner = asyncio.create_task(self.run(), name="idle-compaction")

    async def close(self) -> None:
        if self.runner:
            self.runner.cancel()
            await asyncio.gather(self.runner, return_exceptions=True)

    async def run(self) -> None:
        # Give runner assignment reconciliation a moment to populate owned agents.
        await asyncio.sleep(min(5.0, self.poll_seconds))
        while True:
            try:
                await self.poll_once()
            except ServiceError as error:
                logger.warning("Idle compaction poll failed: %s", error)
            except Exception:
                logger.exception("Unexpected error in idle compaction loop")
            await asyncio.sleep(self.poll_seconds)

    async def _resolve_agent_keys(self) -> list[str]:
        if self._agent_keys is not None:
            return list(self._agent_keys())
        raw_agents = await self.services.json_request("Tsuchi", "GET", "/api/agents")
        if not isinstance(raw_agents, list):
            return []
        return [
            item["key"]
            for item in raw_agents
            if isinstance(item, dict) and isinstance(item.get("key"), str)
        ]

    async def poll_once(self, now: datetime | None = None) -> int:
        current_time = now or datetime.now(UTC)
        status = await self.services.inference_status()
        if status.state != "ready":
            return 0

        agent_keys = await self._resolve_agent_keys()
        if not agent_keys:
            return 0

        raw_runs = await self.services.json_request("Tsuchi", "GET", "/api/agent-runs")
        active_runs = (
            [AgentRun.model_validate(item) for item in raw_runs]
            if isinstance(raw_runs, list)
            else []
        )
        active_conversation_ids = {run.conversation_id for run in active_runs}
        active_agent_keys = {run.agent_key for run in active_runs}

        compacted_count = 0
        for agent_key in agent_keys:
            if agent_key in active_agent_keys:
                continue
            try:
                conversations = await self.services.conversations.all(
                    agent_key, ResourceKind.CONVERSATION
                )
            except ServiceError as error:
                logger.warning(
                    "Idle compaction could not list conversations for %s: %s",
                    agent_key,
                    error,
                )
                continue
            if not conversations:
                continue
            conversation = conversations[0]
            if conversation.id in active_conversation_ids or not conversation.messages:
                continue
            existing_summary, uncompacted = split_uncompacted(
                conversation.messages,
                conversation.compacted_summary,
                conversation.compacted_through_message_id,
            )
            if not uncompacted:
                continue
            latest_message_at = max(
                message.created_at for message in conversation.messages
            )
            if current_time - latest_message_at < self.idle_delay:
                continue
            try:
                new_summary = await generate_rolling_summary(
                    self.inference,
                    agent_key=agent_key,
                    conversation_id=conversation.id,
                    existing_summary=existing_summary,
                    messages=uncompacted,
                    model_id=status.model_id,
                    context_limit=getattr(status, "context_limit", None),
                    priority="background",
                )
                now = datetime.now(UTC)
                snapshot_kwargs: dict[str, object] = {}
                if hasattr(self.services, "list_resource_kinds"):
                    fresh_snapshot, _ = await build_core_memories_snapshot(
                        self.services, agent_key
                    )
                    snapshot_kwargs = {
                        "core_memories_snapshot": fresh_snapshot,
                        "core_memories_refreshed_at": now,
                    }
                await persist_compaction(
                    self.services,
                    agent_key=agent_key,
                    conversation_id=conversation.id,
                    expected_version=conversation.version,
                    compacted_summary=new_summary,
                    compacted_through_message_id=uncompacted[-1].id,
                    compacted_at=now,
                    require_idle=True,
                    **snapshot_kwargs,
                )
                compacted_count += 1
                logger.info(
                    "Idle auto-compacted conversation agent=%s conversation=%s through=%s messages=%d",
                    agent_key,
                    conversation.id,
                    uncompacted[-1].id,
                    len(uncompacted),
                )
            except ServiceError as error:
                if error.status == 409:
                    logger.info(
                        "Skipped idle compaction for agent=%s conversation=%s due to concurrent update or active run",
                        agent_key,
                        conversation.id,
                    )
                else:
                    logger.warning(
                        "Idle compaction failed for agent=%s conversation=%s: %s",
                        agent_key,
                        conversation.id,
                        error,
                    )
            except (ValidationError, OSError) as error:
                logger.warning(
                    "Idle compaction failed for agent=%s conversation=%s: %s",
                    agent_key,
                    conversation.id,
                    error,
                )
        return compacted_count
