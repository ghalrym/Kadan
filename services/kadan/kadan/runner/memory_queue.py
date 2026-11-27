"""Decision-gated background memory and onboarding persona extraction."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from ..api.config import ServiceError
from ..api.memory_context import persist_memory_state
from ..api.pydantic_models.agent_runs import AgentRun
from ..api.pydantic_models.resources import Message, Resource, ResourceKind
from ..api.services import parse_resource
from .agent_inference import AgentInference

if TYPE_CHECKING:
    from ..api.services import LocalServices

logger = logging.getLogger(__name__)

MEMORY_QUEUE_POLL_SECONDS = 25.0
MEMORY_QUEUE_IDLE_DELAY = timedelta(seconds=25)
DECISION_THRESHOLD = 0.55
MAX_DECISION_STATE_CHARS = 5500
MAX_SLICE_MESSAGES = 24
MAX_EXISTING_MEMORIES_IN_PROMPT = 40
MAX_EXTRACTED_MEMORIES_PER_BATCH = 6

CONVERSATION_ONLY_FIELDS = {
    "id",
    "agent_key",
    "version",
    "created_at",
    "updated_at",
    "deleted_at",
    "compacted_summary",
    "compacted_through_message_id",
    "compacted_at",
    "memory_extracted_through_message_id",
    "core_memories_snapshot",
    "core_memories_refreshed_at",
}

EXTRACTION_SYSTEM_PROMPT = (
    "You extract durable memories (and optional persona updates) from a recent "
    "conversation slice between a user and their personal AI assistant.\n"
    "Output ONLY a valid JSON object matching this exact schema (no markdown fences, "
    "no preamble, no extra text):\n"
    "{\n"
    '  "memories": [\n'
    "    {\n"
    '      "action": "create" | "update",\n'
    '      "memory_id": "<UUID from existing memories when action is update, else null>",\n'
    '      "title": "<concise specific title>",\n'
    '      "markdown": "<concise factual Markdown summary>",\n'
    '      "tags": ["<lowercase tags; include \'core\' for always-needed facts>"]\n'
    "    }\n"
    "  ],\n"
    '  "persona_markdown": "<full updated persona Markdown, or null if no change>"\n'
    "}\n\n"
    "Rules:\n"
    "- Extract ONLY durable facts explicitly stated or confirmed by the user "
    "(name, identity, standing preferences, goals, ongoing projects, relationships, "
    "constraints, corrections). Never invent facts or save trivial small talk.\n"
    "- Check existing memories first: when a fact refines or updates an existing "
    "memory, use action='update' with its exact memory_id instead of creating a duplicate.\n"
    "- Tag core always-needed facts (user's name, core identity, standing preferences, "
    "major ongoing context) with 'core'.\n"
    "- Only return non-null persona_markdown when persona updates were requested AND "
    "the user answered an onboarding TODO or explicitly asked to change how the agent "
    "acts, speaks, or looks. Preserve all existing persona details, fill in answered TODOs, "
    "and remove the '# Onboarding' section once every TODO is filled in.\n"
    "- If nothing warrants saving, return {\"memories\": [], \"persona_markdown\": null}."
)


class ExtractedMemoryItem(BaseModel):
    action: str = "create"
    memory_id: UUID | None = None
    title: str = Field(min_length=1, max_length=500)
    markdown: str = Field(default="", max_length=16000)
    tags: list[str] = Field(default_factory=list, max_length=50)


class ExtractionPayload(BaseModel):
    memories: list[ExtractedMemoryItem] = Field(default_factory=list)
    persona_markdown: str | None = None


def unextracted_messages(
    messages: Sequence[Message],
    extracted_through_message_id: UUID | None,
) -> list[Message]:
    """Return user/assistant messages strictly after the durable memory cursor."""
    if extracted_through_message_id is not None:
        anchor_index = next(
            (
                index
                for index, message in enumerate(messages)
                if message.id == extracted_through_message_id
            ),
            None,
        )
        if anchor_index is not None:
            return [
                message
                for message in messages[anchor_index + 1 :]
                if message.role in {"user", "assistant"}
            ]
    return [
        message for message in messages if message.role in {"user", "assistant"}
    ]


def format_turn_slice(
    messages: Sequence[Message], *, max_chars: int = MAX_DECISION_STATE_CHARS
) -> str:
    """Format the new conversation slice concisely for decision gating and extraction."""
    selected = list(messages[-MAX_SLICE_MESSAGES:])
    lines: list[str] = []
    for message in selected:
        stamp = message.created_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
        speaker = "User" if message.role == "user" else "Assistant"
        content = message.content.strip()
        if len(content) > 1200:
            content = content[:1199].rstrip() + "…"
        lines.append(f"[{stamp}] {speaker}: {content}")
    joined = "\n\n".join(lines)
    if len(joined) <= max_chars:
        return joined
    return "…" + joined[-(max_chars - 1) :]


def build_decision_questions(
    *, include_persona_gate: bool
) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = [
        {
            "key": "should_save_memory",
            "type": "Noul",
            "instructions": (
                "Did the user share any durable personal fact, name, preference, goal, "
                "project detail, relationship, constraint, or correction worth "
                "remembering long-term?"
            ),
            "trueWhen": (
                "The user stated a concrete personal fact, their name, preference, "
                "goal, ongoing project, relationship, constraint, or corrected a "
                "mistake that should be remembered in future conversations."
            ),
            "falseWhen": (
                "Small talk, greetings, one-off factual questions, transient requests, "
                "or assistant messages without new durable facts from the user."
            ),
        },
        {
            "key": "is_core_memory",
            "type": "Noul",
            "instructions": (
                "Is any durable fact shared by the user a core fact the agent should "
                "always know in every conversation (such as the user's name, identity, "
                "standing preferences, or major ongoing context)?"
            ),
            "trueWhen": (
                "The user shared their name, core identity, standing communication "
                "or work preferences, or major ongoing life/work context."
            ),
            "falseWhen": (
                "No durable fact was shared, or the fact is minor/situational rather "
                "than always-needed core context."
            ),
        },
    ]
    if include_persona_gate:
        questions.append(
            {
                "key": "should_update_persona",
                "type": "Noul",
                "instructions": (
                    "Did the user answer an onboarding TODO or ask to change how the "
                    "agent acts, speaks, is named, or looks?"
                ),
                "trueWhen": (
                    "The user provided an answer to an unfilled onboarding TODO "
                    "(agent name, user's name, vibe/tone, role, communication style, "
                    "or appearance) or asked to adjust the agent's personality, style, "
                    "or persona."
                ),
                "falseWhen": (
                    "The user did not answer an onboarding TODO and did not request "
                    "any change to how the agent behaves or looks."
                ),
            }
        )
    return questions


def parse_extraction_json(raw: str) -> ExtractionPayload:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    parsed = json.loads(text)
    return ExtractionPayload.model_validate(parsed)


class MemoryQueue:
    """Decision-gated background worker that extracts memories and onboarding updates."""

    def __init__(
        self,
        services: LocalServices,
        *,
        agent_keys: Callable[[], Sequence[str]] | None = None,
        inference: AgentInference | None = None,
        poll_seconds: float = MEMORY_QUEUE_POLL_SECONDS,
        idle_delay: timedelta = MEMORY_QUEUE_IDLE_DELAY,
        decision_threshold: float = DECISION_THRESHOLD,
    ):
        self.services = services
        self._agent_keys = agent_keys
        self.inference = inference or AgentInference(
            services.client, services.settings.hikari_url
        )
        self.poll_seconds = poll_seconds
        self.idle_delay = idle_delay
        self.decision_threshold = decision_threshold
        self.wakeup = asyncio.Event()
        self.runner: asyncio.Task | None = None
        self._active_work: asyncio.Task | None = None
        self._next_idle_wait: float | None = None

    def set_agent_keys_provider(
        self, provider: Callable[[], Sequence[str]]
    ) -> None:
        self._agent_keys = provider

    def preempt(self) -> bool:
        """Cancel any in-flight background work so interactive runs get the GPU immediately."""
        task = self._active_work
        if task is not None and not task.done():
            logger.info("Preempting in-flight memory queue work for interactive run")
            task.cancel()
            return True
        return False

    def start(self) -> None:
        self.runner = asyncio.create_task(self.run(), name="memory-queue")

    async def close(self) -> None:
        self.preempt()
        if self.runner:
            self.runner.cancel()
            await asyncio.gather(self.runner, return_exceptions=True)

    async def run(self) -> None:
        await asyncio.sleep(min(2.0, self.poll_seconds))
        while True:
            self.wakeup.clear()
            self._next_idle_wait = None
            try:
                await self.poll_once()
            except ServiceError as error:
                logger.warning("Memory queue poll failed: %s", error)
            except Exception:
                logger.exception("Unexpected error in memory queue loop")
            timeout = self.poll_seconds
            if self._next_idle_wait is not None:
                timeout = min(timeout, max(0.05, self._next_idle_wait))
            try:
                await asyncio.wait_for(self.wakeup.wait(), timeout=timeout)
            except TimeoutError:
                pass

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

    async def _active_run_sets(self) -> tuple[set[str], set[UUID]]:
        raw_runs = await self.services.json_request("Tsuchi", "GET", "/api/agent-runs")
        active_runs = (
            [AgentRun.model_validate(item) for item in raw_runs]
            if isinstance(raw_runs, list)
            else []
        )
        return (
            {run.agent_key for run in active_runs},
            {run.conversation_id for run in active_runs},
        )

    async def poll_once(self, now: datetime | None = None) -> int:
        current_time = now or datetime.now(UTC)
        self._next_idle_wait = None
        status = await self.services.inference_status()
        if status.state != "ready":
            return 0
        use_local_decisions = status.model_id is not None

        agent_keys = await self._resolve_agent_keys()
        if not agent_keys:
            return 0

        active_agent_keys, active_conversation_ids = await self._active_run_sets()
        processed_count = 0
        soonest_wait: float | None = None

        for agent_key in agent_keys:
            if agent_key in active_agent_keys:
                continue
            try:
                conversations = await self.services.conversations.all(
                    agent_key, ResourceKind.CONVERSATION
                )
            except ServiceError as error:
                logger.warning(
                    "Memory queue could not list conversations for %s: %s",
                    agent_key,
                    error,
                )
                continue
            for conversation in conversations:
                if conversation.id in active_conversation_ids or not conversation.messages:
                    continue
                new_messages = unextracted_messages(
                    conversation.messages,
                    conversation.memory_extracted_through_message_id,
                )
                if not new_messages:
                    continue
                latest_message_at = conversation.messages[-1].created_at
                age = current_time - latest_message_at
                if age < self.idle_delay:
                    remaining = (self.idle_delay - age).total_seconds()
                    soonest_wait = (
                        remaining
                        if soonest_wait is None
                        else min(soonest_wait, remaining)
                    )
                    continue
                work = asyncio.create_task(
                    self._process_conversation(
                        agent_key=agent_key,
                        conversation=conversation,
                        new_messages=new_messages,
                        model_id=status.model_id,
                        use_local_decisions=use_local_decisions,
                    ),
                    name=f"memory-queue-work-{agent_key}",
                )
                self._active_work = work
                try:
                    did_process = await work
                    if did_process:
                        processed_count += 1
                except asyncio.CancelledError:
                    current_task = asyncio.current_task()
                    if current_task is not None and current_task.cancelling():
                        if not work.done():
                            work.cancel()
                            await asyncio.gather(work, return_exceptions=True)
                        raise
                    logger.info(
                        "Memory extraction preempted for agent=%s conversation=%s; will retry when idle",
                        agent_key,
                        conversation.id,
                    )
                    self._next_idle_wait = soonest_wait
                    return processed_count
                except ServiceError as error:
                    if error.status == 409:
                        logger.info(
                            "Skipped memory extraction for agent=%s conversation=%s due to active run or conflict",
                            agent_key,
                            conversation.id,
                        )
                    else:
                        logger.warning(
                            "Memory extraction failed for agent=%s conversation=%s: %s",
                            agent_key,
                            conversation.id,
                            error,
                        )
                except (ValidationError, ValueError, OSError) as error:
                    logger.warning(
                        "Memory extraction failed for agent=%s conversation=%s: %s",
                        agent_key,
                        conversation.id,
                        error,
                    )
                finally:
                    if self._active_work is work:
                        self._active_work = None
        self._next_idle_wait = soonest_wait
        return processed_count

    async def _process_conversation(
        self,
        *,
        agent_key: str,
        conversation: Resource,
        new_messages: list[Message],
        model_id: str | None,
        use_local_decisions: bool,
    ) -> bool:
        last_message_id = new_messages[-1].id
        # If the unprocessed slice has no user messages (e.g. initial onboarding greeting),
        # advance the cursor directly without spending inference.
        if not any(message.role == "user" for message in new_messages):
            await persist_memory_state(
                self.services,
                agent_key=agent_key,
                conversation_id=conversation.id,
                memory_extracted_through_message_id=last_message_id,
                set_cursor=True,
                require_idle=True,
            )
            return True

        persona = await self.services.agent_persona(agent_key)
        persona_has_todos = persona is not None and "TODO" in persona.markdown
        slice_text = format_turn_slice(new_messages)

        decision_state = slice_text
        if persona_has_todos:
            todo_lines = [
                line.strip()
                for line in persona.markdown.splitlines()
                if "TODO" in line
            ][:8]
            if todo_lines:
                decision_state = (
                    "Unfilled persona onboarding TODOs:\n"
                    + "\n".join(todo_lines)
                    + "\n\nRecent conversation:\n"
                    + slice_text
                )[:MAX_DECISION_STATE_CHARS]

        include_persona_gate = persona_has_todos or persona is not None
        questions = build_decision_questions(include_persona_gate=include_persona_gate)
        scores = await self.inference.evaluate_decisions(
            decision_state,
            questions,
            use_local_decisions=use_local_decisions,
            model_id=model_id,
        )
        save_memory = scores.get("should_save_memory", 0.0) >= self.decision_threshold
        is_core = scores.get("is_core_memory", 0.0) >= self.decision_threshold
        update_persona = (
            include_persona_gate
            and scores.get("should_update_persona", 0.0) >= self.decision_threshold
        )

        if not save_memory and not update_persona:
            logger.info(
                "Memory gate NO agent=%s conversation=%s scores=%s; advancing cursor",
                agent_key,
                conversation.id,
                {k: round(v, 3) for k, v in scores.items()},
            )
            await persist_memory_state(
                self.services,
                agent_key=agent_key,
                conversation_id=conversation.id,
                memory_extracted_through_message_id=last_message_id,
                set_cursor=True,
                require_idle=True,
            )
            return True

        existing_memories = await self.services.list_resource_kinds(
            agent_key, (ResourceKind.MEMORY,), limit=100
        )
        extraction = await self._extract_updates(
            slice_text=slice_text,
            existing_memories=existing_memories,
            persona=persona if update_persona else None,
            save_memory=save_memory,
            is_core_hint=is_core,
            update_persona=update_persona,
            model_id=model_id,
        )

        # Recheck idle state before mutating memories/persona so we never race an active run.
        active_agents, active_convs = await self._active_run_sets()
        if agent_key in active_agents or conversation.id in active_convs:
            raise ServiceError(
                "Conversation became active during memory extraction.",
                409,
                "active_run",
            )

        if save_memory and extraction.memories:
            await self._apply_memory_updates(
                agent_key,
                existing_memories,
                extraction.memories[:MAX_EXTRACTED_MEMORIES_PER_BATCH],
                default_core=is_core,
            )

        if (
            update_persona
            and extraction.persona_markdown
            and extraction.persona_markdown.strip()
        ):
            new_markdown = extraction.persona_markdown.strip()
            if persona is None or new_markdown != persona.markdown.strip():
                try:
                    await self.services.save_agent_persona(
                        agent_key,
                        markdown=new_markdown,
                        expected_version=persona.version if persona else None,
                    )
                except ServiceError as error:
                    if error.status == 409:
                        logger.info(
                            "Skipped concurrent persona update during memory extraction agent=%s",
                            agent_key,
                        )
                    else:
                        raise

        # Advance cursor durably without mutating an already-set core_memories_snapshot.
        await persist_memory_state(
            self.services,
            agent_key=agent_key,
            conversation_id=conversation.id,
            memory_extracted_through_message_id=last_message_id,
            set_cursor=True,
            require_idle=True,
        )
        logger.info(
            "Memory extraction completed agent=%s conversation=%s memories=%d persona_updated=%s",
            agent_key,
            conversation.id,
            len(extraction.memories) if save_memory else 0,
            bool(update_persona and extraction.persona_markdown),
        )
        return True

    async def _extract_updates(
        self,
        *,
        slice_text: str,
        existing_memories: Sequence[Resource],
        persona: Resource | None,
        save_memory: bool,
        is_core_hint: bool,
        update_persona: bool,
        model_id: str | None,
    ) -> ExtractionPayload:
        sections: list[str] = [
            f"Tasks: save_memory={save_memory}, is_core_hint={is_core_hint}, update_persona={update_persona}"
        ]
        if existing_memories:
            mem_lines = []
            for item in existing_memories[:MAX_EXISTING_MEMORIES_IN_PROMPT]:
                tags = ", ".join(item.tags) if item.tags else "none"
                excerpt = " ".join(item.markdown.split())[:140]
                mem_lines.append(
                    f"- id={item.id} | title={item.title!r} | tags=[{tags}] | {excerpt}"
                )
            sections.append("Existing active memories:\n" + "\n".join(mem_lines))
        else:
            sections.append("Existing active memories: (none)")
        if update_persona and persona is not None:
            sections.append(f"Current persona Markdown:\n{persona.markdown}")
        sections.append(f"New conversation messages:\n{slice_text}")

        raw = await self.inference.complete_text(
            [
                SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(content="\n\n".join(sections)),
            ],
            model_id,
            conversation_key=None,
            priority="background",
            reuse_prefix=False,
            max_tokens=1536,
            reasoning_tokens=256,
            reasoning_effort="low",
        )
        return parse_extraction_json(raw)

    async def _apply_memory_updates(
        self,
        agent_key: str,
        existing_memories: Sequence[Resource],
        items: Sequence[ExtractedMemoryItem],
        *,
        default_core: bool,
    ) -> None:
        by_id: dict[UUID, Resource] = {item.id: item for item in existing_memories}
        by_title: dict[str, Resource] = {
            item.title.strip().casefold(): item for item in existing_memories
        }
        base_path = f"/api/agents/{agent_key}/resources"
        for item in items:
            title = item.title.strip()
            markdown = item.markdown.strip()
            if not title or not markdown:
                continue
            tags = list(dict.fromkeys(t.strip().lower() for t in item.tags if t.strip()))
            if default_core and "core" not in tags:
                tags.append("core")

            target: Resource | None = None
            if item.action == "update" and item.memory_id in by_id:
                target = by_id[item.memory_id]
            elif title.casefold() in by_title:
                target = by_title[title.casefold()]

            if target is not None:
                merged_tags = list(
                    dict.fromkeys([*(t.strip().lower() for t in target.tags if t.strip()), *tags])
                )
                if (
                    target.title.strip() == title
                    and target.markdown.strip() == markdown
                    and [t.strip().lower() for t in target.tags] == merged_tags
                ):
                    continue
                content = target.model_dump(mode="json", exclude=CONVERSATION_ONLY_FIELDS)
                content.update(
                    title=title,
                    markdown=markdown,
                    tags=merged_tags,
                    expected_version=target.version,
                )
                try:
                    updated = parse_resource(
                        await self.services.json_request(
                            "Tsuchi",
                            "PUT",
                            f"{base_path}/{target.id}",
                            params={"preserve_memory_snapshot": "true"},
                            json=content,
                        )
                    )
                    by_id[updated.id] = updated
                    by_title[updated.title.strip().casefold()] = updated
                except ServiceError as error:
                    if error.status != 409:
                        raise
            else:
                created = parse_resource(
                    await self.services.json_request(
                        "Tsuchi",
                        "POST",
                        base_path,
                        params={"preserve_memory_snapshot": "true"},
                        json={
                            "kind": "memory",
                            "title": title,
                            "markdown": markdown,
                            "tags": tags,
                        },
                    )
                )
                by_id[created.id] = created
                by_title[created.title.strip().casefold()] = created
