"""Unit tests for KV-cache-safe hybrid memory, tail recall, MemoryQueue, voice call injection, and single-call persona edits."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import httpx
import pytest

from kadan.api.config import ServiceError, Settings
from kadan.api.memory_context import (
    extract_recall_terms,
    format_core_memories,
    select_core_memories,
)
from kadan.api.pydantic_models.agent_runs import (
    AgentRun,
    RunMessage,
    RunObservation,
)
from kadan.api.pydantic_models.resources import Agent, Message, Resource, ResourceKind
from kadan.api.services import STARTER_PERSONA_MARKDOWN, LocalServices
from kadan.runner.agent_runs import AgentRuns, RunConversation
from kadan.runner.compaction import IdleCompactor
from kadan.runner.config import RunnerSettings
from kadan.runner.memory_queue import MEMORY_QUEUE_IDLE_DELAY, MemoryQueue
from kadan.runner.service import AgentRunner
from kadan.runner.tools.persona import UpdatePersonaArguments


def make_memory(
    title: str,
    markdown: str,
    *,
    tags: list[str] | None = None,
    updated_at: datetime | None = None,
    pinned: bool = False,
    memory_id: UUID | None = None,
    version: int = 1,
) -> Resource:
    ts = updated_at or datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    return Resource(
        id=memory_id or uuid4(),
        agent_key="alpha",
        kind=ResourceKind.MEMORY,
        title=title,
        markdown=markdown,
        tags=tags or [],
        pinned=pinned,
        version=version,
        created_at=ts,
        updated_at=ts,
    )


def test_core_memories_selection_prefers_core_tag_and_falls_back_under_five():
    base = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    m1 = make_memory("User Name", "Andrew", tags=["CORE"], updated_at=base)
    m2 = make_memory("Greenhouse", "Cedar frame", tags=["project"], updated_at=base + timedelta(minutes=1))

    selected = select_core_memories([m1, m2])
    assert [item.id for item in selected] == [m1.id]
    formatted = format_core_memories([m1, m2])
    assert "**User Name**: Andrew" in formatted
    assert "Greenhouse" not in formatted

    # Fallback when <= 5 memories and none are tagged 'core'.
    untagged = [
        make_memory(f"Fact {i}", f"Detail {i}", updated_at=base + timedelta(minutes=i))
        for i in range(4)
    ]
    assert len(select_core_memories(untagged)) == 4

    # When > 5 untagged memories exist and none are tagged 'core', fallback does not dump everything.
    many_untagged = [
        make_memory(f"Fact {i}", f"Detail {i}", updated_at=base + timedelta(minutes=i))
        for i in range(6)
    ]
    assert select_core_memories(many_untagged) == []


def test_extract_recall_terms_ignores_stop_words_and_greetings():
    assert extract_recall_terms("Hey there, thanks!") == []
    terms = extract_recall_terms(
        "Can you check my cedar greenhouse irrigation blueprint from last week?"
    )
    assert "irrigation" in terms
    assert "greenhouse" in terms
    assert "blueprint" in terms


def test_starter_persona_includes_users_name_and_single_call_onboarding():
    assert "- **User's Name:** TODO" in STARTER_PERSONA_MARKDOWN
    assert "Core Memories" in STARTER_PERSONA_MARKDOWN
    assert '"core"' in STARTER_PERSONA_MARKDOWN
    assert "only once" in STARTER_PERSONA_MARKDOWN
    assert "Do **not** call `create_memory` on every turn" in STARTER_PERSONA_MARKDOWN
    assert "Do **not** call `update_persona` on every individual onboarding turn" in STARTER_PERSONA_MARKDOWN


def test_single_call_update_persona_resolves_current_version_when_omitted():
    args = UpdatePersonaArguments.model_validate({"markdown": "Updated persona."})
    assert args.expected_version is None

    async def check():
        existing = Resource(
            id=uuid4(),
            agent_key="alpha",
            kind=ResourceKind.PERSONA,
            title="Alpha",
            markdown="Old persona.",
            version=4,
            created_at=datetime(2026, 11, 26, 12, 0, tzinfo=UTC),
            updated_at=datetime(2026, 11, 26, 12, 0, tzinfo=UTC),
        )
        saved_calls = []

        async def save_persona(key, *, markdown, expected_version, title=None):
            saved_calls.append((key, markdown, expected_version, title))
            return existing.model_copy(update={"markdown": markdown, "version": 5})

        services = SimpleNamespace(
            agent_persona=AsyncMock(return_value=existing),
            save_agent_persona=save_persona,
        )
        runs = SimpleNamespace(services=services, owned=AsyncMock())
        run = SimpleNamespace(id=uuid4(), agent_key="alpha", transcript=[])
        conv = RunConversation(runs, run)
        result = await conv.update_persona(args)
        assert result["persona"]["version"] == 5
        assert saved_calls == [("alpha", "Updated persona.", 4, None)]
        assert runs.owned.await_count == 2

    asyncio.run(check())


def test_call_instruction_injects_core_memories_and_trims_memories_before_persona():
    async def check():
        base = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
        core = make_memory(
            "User Name", "Andrew", tags=["core"], updated_at=base + timedelta(minutes=5)
        )
        big_core = make_memory(
            "Preferences", "detail " * 800, tags=["core"], updated_at=base
        )

        def handler(request):
            assert request.url.path == "/api/agents/alpha/resources"
            return httpx.Response(
                200,
                json=[
                    core.model_dump(mode="json"),
                    big_core.model_dump(mode="json"),
                ],
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = LocalServices(client, Settings(_env_file=None))
            persona = Resource(
                id=uuid4(),
                agent_key="alpha",
                kind=ResourceKind.PERSONA,
                title="Lumi",
                markdown="Warm and concise.",
                version=1,
                created_at=datetime(2026, 11, 26, 12, 0, tzinfo=UTC),
                updated_at=datetime(2026, 11, 26, 12, 0, tzinfo=UTC),
            )
            service.agent = AsyncMock(return_value=Agent(key="alpha", name="Lumi"))
            service.agent_persona = AsyncMock(return_value=persona)

            instruction = await service.call_instruction("alpha")
            assert len(instruction) <= 8000
            assert "Warm and concise." in instruction
            assert "**User Name**: Andrew" in instruction

            # With a large persona (~7,600 chars), persona stays intact and memories are trimmed to fit <= 8,000.
            long_persona = persona.model_copy(update={"markdown": "p" * 7400})
            service.agent_persona = AsyncMock(return_value=long_persona)
            tight = await service.call_instruction("alpha")
            assert len(tight) <= 8000
            assert long_persona.markdown in tight
            assert "**User Name**: Andrew" in tight

    asyncio.run(check())


class HybridBoundary:
    """In-memory Tsuchi boundary supporting memories, memory-state, compaction, and persona."""

    def __init__(self, instruction_file, messages, *, memories=None, persona=None):
        now = datetime(2026, 11, 26, 15, 0, tzinfo=UTC)
        self.client = object()  # Non-None marker so memory helpers run.
        self.settings = SimpleNamespace(
            hikari_url="http://unused", agent_instruction_file=str(instruction_file)
        )
        self.images = SimpleNamespace(wakeup=asyncio.Event())
        self.memory_queue = SimpleNamespace(wakeup=asyncio.Event())
        self.conversation = Resource(
            id=uuid4(),
            agent_key="alpha",
            kind=ResourceKind.CONVERSATION,
            title="New chat",
            messages=messages,
            version=len(messages) or 1,
            created_at=now,
            updated_at=now,
        )
        self.memories: list[Resource] = list(memories or [])
        self.persona: Resource | None = persona
        self.active_runs: list[AgentRun] = []
        self.status = SimpleNamespace(
            state="ready", model_id="inkling-2-mini-iq4-nl", context_limit=32768
        )
        self.memory_state_requests: list[dict] = []
        self.compaction_requests: list[dict] = []
        self.memory_writes: list[tuple[str, str, dict, dict]] = []
        self.persona_saves: list[tuple[str, int | None]] = []
        self.conversations = SimpleNamespace(all=self.list_conversations)
        self.controller = AgentRuns(self)

    def make_run(self, trigger_message_id: UUID) -> AgentRun:
        now = datetime(2026, 11, 26, 15, 5, tzinfo=UTC)
        return AgentRun(
            id=uuid4(),
            agent_key="alpha",
            conversation_id=self.conversation.id,
            trigger_message_id=trigger_message_id,
            version=1,
            created_at=now,
            updated_at=now,
        )

    async def list_conversations(self, key, kind):
        return [self.conversation]

    async def resource(self, key, resource_id):
        assert resource_id == self.conversation.id
        return self.conversation

    async def agent(self, key):
        return Agent(key=key, name="Lumi")

    async def agent_persona(self, key):
        return self.persona

    async def save_agent_persona(self, key, *, markdown, expected_version, title=None):
        self.persona_saves.append((markdown, expected_version))
        now = datetime(2026, 11, 26, 15, 10, tzinfo=UTC)
        if self.persona is None:
            self.persona = Resource(
                id=uuid4(),
                agent_key=key,
                kind=ResourceKind.PERSONA,
                title=title or "Lumi",
                markdown=markdown,
                version=1,
                created_at=now,
                updated_at=now,
            )
        else:
            self.persona = self.persona.model_copy(
                update={"markdown": markdown, "version": self.persona.version + 1}
            )
        return self.persona

    async def inference_status(self):
        return self.status

    async def list_resource_kinds(self, key, kinds, *, q="", limit=100, offset=0):
        assert ResourceKind.MEMORY in kinds
        active = [m for m in self.memories if m.deleted_at is None]
        if q:
            needle = q.casefold()
            active = [
                m
                for m in active
                if needle in " ".join([m.title, m.markdown, *m.tags]).casefold()
            ]
        return active[offset : offset + limit]

    async def json_request(self, service, method, path, **kwargs):
        assert service == "Tsuchi"
        body = kwargs.get("json", {})
        params = kwargs.get("params", {})
        if path == "/api/agent-runs" and method == "GET":
            return [run.model_dump(mode="json") for run in self.active_runs]
        if path == f"/api/agents/alpha/resources/{self.conversation.id}/runs" and method == "GET":
            return []
        if path == f"/api/agents/alpha/resources/{self.conversation.id}/memory-state" and method == "PUT":
            self.memory_state_requests.append(body)
            updates = {}
            if "memory_extracted_through_message_id" in body:
                updates["memory_extracted_through_message_id"] = UUID(
                    body["memory_extracted_through_message_id"]
                )
            if "core_memories_snapshot" in body:
                if not (
                    body.get("only_if_snapshot_unset")
                    and self.conversation.core_memories_snapshot is not None
                ):
                    updates["core_memories_snapshot"] = body["core_memories_snapshot"]
                    updates["core_memories_refreshed_at"] = (
                        datetime.fromisoformat(body["core_memories_refreshed_at"])
                        if body.get("core_memories_refreshed_at")
                        else datetime.now(UTC)
                    )
            self.conversation = self.conversation.model_copy(update=updates)
            return self.conversation.model_dump(mode="json")
        if path == f"/api/agents/alpha/resources/{self.conversation.id}/compaction" and method == "PUT":
            self.compaction_requests.append(body)
            updates = {
                "compacted_summary": body["compacted_summary"],
                "compacted_through_message_id": UUID(body["compacted_through_message_id"]),
                "compacted_at": datetime.now(UTC),
                "version": self.conversation.version + 1,
            }
            if "core_memories_snapshot" in body:
                updates["core_memories_snapshot"] = body["core_memories_snapshot"]
                updates["core_memories_refreshed_at"] = datetime.now(UTC)
            self.conversation = self.conversation.model_copy(update=updates)
            return self.conversation.model_dump(mode="json")
        if path == "/api/agents/alpha/resources" and method == "POST":
            self.memory_writes.append(("POST", path, dict(params), dict(body)))
            created = make_memory(
                body["title"], body["markdown"], tags=body.get("tags", [])
            )
            self.memories.append(created)
            return created.model_dump(mode="json")
        if path.startswith("/api/agents/alpha/resources/") and method == "PUT":
            self.memory_writes.append(("PUT", path, dict(params), dict(body)))
            target_id = UUID(path.rsplit("/", 1)[-1])
            for idx, item in enumerate(self.memories):
                if item.id == target_id:
                    updated = item.model_copy(
                        update={
                            "title": body["title"],
                            "markdown": body["markdown"],
                            "tags": body.get("tags", item.tags),
                            "version": item.version + 1,
                        }
                    )
                    self.memories[idx] = updated
                    return updated.model_dump(mode="json")
        raise AssertionError(f"Unexpected {method} {path}")


def test_epoch_frozen_core_memories_and_tail_archival_recall(tmp_path):
    instructions = tmp_path / "AGENT.md"
    instructions.write_text("Base instructions.")
    base = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    m1 = Message(id=uuid4(), role="user", content="Tell me about my greenhouse irrigation plan.", created_at=base)
    core_mem = make_memory("User Name", "Andrew", tags=["core"])
    archival_mem = make_memory(
        "Greenhouse Irrigation",
        "Drip lines connected to rainwater barrel.",
        tags=["project"],
    )
    boundary = HybridBoundary(instructions, [m1], memories=[core_mem, archival_mem])

    # Turn 1: builds and freezes core_memories_snapshot, recalls archival at tail.
    ctx1 = asyncio.run(boundary.controller.context(boundary.make_run(m1.id), limit=20000))
    system1 = ctx1[0].content
    assert "**User Name**: Andrew" in system1
    assert "Greenhouse Irrigation" not in system1
    assert boundary.conversation.core_memories_snapshot is not None
    assert len(boundary.memory_state_requests) == 1
    assert boundary.memory_state_requests[0]["only_if_snapshot_unset"] is True
    # Archival recall is appended as a tail RunObservation, never in system.
    assert isinstance(ctx1[-1], RunObservation)
    assert ctx1[-1].name == "recalled_memories"
    assert ctx1[-1].data["memories"][0]["title"] == "Greenhouse Irrigation"

    # Background adds a new core memory mid-epoch WITHOUT clearing snapshot.
    boundary.memories.append(make_memory("Coffee", "Loves pour-over coffee", tags=["core"]))
    m2 = Message(id=uuid4(), role="user", content="Thanks!", created_at=base + timedelta(minutes=2))
    boundary.conversation = boundary.conversation.model_copy(
        update={"messages": [*boundary.conversation.messages, m2]}
    )

    # Turn 2: reuses frozen snapshot byte-identically (KV cache stays hot!) and skips recall on greeting.
    ctx2 = asyncio.run(boundary.controller.context(boundary.make_run(m2.id), limit=20000))
    assert ctx2[0].content == system1
    assert "pour-over coffee" not in ctx2[0].content
    assert not any(
        isinstance(item, RunObservation) and item.name == "recalled_memories"
        for item in ctx2
    )

    # Idle compaction refreshes core_memories_snapshot atomically with compacted_summary!
    fake_inf = SimpleNamespace(
        complete_text=AsyncMock(return_value="Rolling summary of greenhouse chat.")
    )
    compactor = IdleCompactor(
        boundary, agent_keys=lambda: ["alpha"], inference=fake_inf
    )
    compacted = asyncio.run(compactor.poll_once(now=base + timedelta(hours=2)))
    assert compacted == 1
    assert "pour-over coffee" in (boundary.conversation.core_memories_snapshot or "")
    assert "core_memories_snapshot" in boundary.compaction_requests[0]

    # Turn 3 in the new epoch sees the refreshed Core Memories + summary!
    m3 = Message(id=uuid4(), role="user", content="Good morning!", created_at=base + timedelta(hours=3))
    boundary.conversation = boundary.conversation.model_copy(
        update={"messages": [*boundary.conversation.messages, m3]}
    )
    ctx3 = asyncio.run(boundary.controller.context(boundary.make_run(m3.id), limit=20000))
    assert "pour-over coffee" in ctx3[0].content
    assert "Rolling summary of greenhouse chat." in ctx3[0].content
    # Ordering: persona -> Core Memories -> compacted_summary.
    assert ctx3[0].content.index("Core memories") < ctx3[0].content.index("Summary of earlier conversation")


def test_memory_queue_fast_gate_no_spends_zero_generation_tokens(tmp_path):
    instructions = tmp_path / "AGENT.md"
    instructions.write_text("Base instructions.")
    base = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    messages = [
        Message(id=uuid4(), role="user", content="Hey, what's 2 + 2?", created_at=base),
        Message(id=uuid4(), role="assistant", content="4!", created_at=base + timedelta(seconds=5)),
    ]
    boundary = HybridBoundary(instructions, messages)
    fake_inf = SimpleNamespace(
        evaluate_decisions=AsyncMock(
            return_value={"should_save_memory": 0.12, "is_core_memory": 0.05}
        ),
        complete_text=AsyncMock(side_effect=AssertionError("Should not generate!")),
    )
    queue = MemoryQueue(boundary, agent_keys=lambda: ["alpha"], inference=fake_inf)

    processed = asyncio.run(queue.poll_once())
    assert processed == 1
    fake_inf.evaluate_decisions.assert_awaited_once()
    fake_inf.complete_text.assert_not_awaited()
    assert boundary.conversation.memory_extracted_through_message_id == messages[-1].id
    assert boundary.memory_writes == []

    # Second poll sees nothing new and makes zero calls.
    second = asyncio.run(queue.poll_once())
    assert second == 0
    assert fake_inf.evaluate_decisions.await_count == 1


def test_memory_queue_gate_yes_extracts_memories_updates_persona_and_preserves_snapshot(
    tmp_path,
):
    instructions = tmp_path / "AGENT.md"
    instructions.write_text("Base instructions.")
    base = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    messages = [
        Message(
            id=uuid4(),
            role="user",
            content="Call yourself Lumi! My name is Andrew and I like warm, concise replies.",
            created_at=base,
        ),
        Message(
            id=uuid4(),
            role="assistant",
            content="Nice to meet you, Andrew! I'm Lumi.",
            created_at=base + timedelta(seconds=5),
        ),
    ]
    existing_mem = make_memory("Communication Style", "Casual tone", tags=["core"])
    persona = Resource(
        id=uuid4(),
        agent_key="alpha",
        kind=ResourceKind.PERSONA,
        title="Kadan Agent",
        markdown=STARTER_PERSONA_MARKDOWN,
        version=1,
        created_at=base,
        updated_at=base,
    )
    boundary = HybridBoundary(
        instructions, messages, memories=[existing_mem], persona=persona
    )
    # Pre-freeze an epoch snapshot so we verify background extraction does NOT clobber it.
    boundary.conversation = boundary.conversation.model_copy(
        update={
            "core_memories_snapshot": "Core memories:\n- **Communication Style**: Casual tone",
            "core_memories_refreshed_at": base,
        }
    )

    extraction_json = json.dumps(
        {
            "memories": [
                {
                    "action": "create",
                    "memory_id": None,
                    "title": "User's Name",
                    "markdown": "The user's name is Andrew.",
                    "tags": ["core"],
                },
                {
                    "action": "update",
                    "memory_id": str(existing_mem.id),
                    "title": "Communication Style",
                    "markdown": "Prefers warm, concise replies.",
                    "tags": ["core"],
                },
            ],
            "persona_markdown": "# Identity\n- **Name:** Lumi\n- **User's Name:** Andrew\n- **Communication Style:** Warm and concise",
        }
    )
    fake_inf = SimpleNamespace(
        evaluate_decisions=AsyncMock(
            return_value={
                "should_save_memory": 0.92,
                "is_core_memory": 0.88,
                "should_update_persona": 0.81,
            }
        ),
        complete_text=AsyncMock(return_value=extraction_json),
    )
    queue = MemoryQueue(
        boundary,
        agent_keys=lambda: ["alpha"],
        inference=fake_inf,
        idle_delay=timedelta(0),
    )

    processed = asyncio.run(queue.poll_once())
    assert processed == 1
    # Verify extraction call was KV-cache safe and token-capped.
    call_kwargs = fake_inf.complete_text.await_args.kwargs
    assert call_kwargs["conversation_key"] is None
    assert call_kwargs["reuse_prefix"] is False
    assert call_kwargs["priority"] == "background"
    assert call_kwargs["reasoning_effort"] == "low"
    assert call_kwargs["reasoning_tokens"] == 256
    assert call_kwargs["max_tokens"] == 1536

    # Verify both memory writes passed preserve_memory_snapshot=true.
    assert len(boundary.memory_writes) == 2
    assert boundary.memory_writes[0][0] == "POST"
    assert boundary.memory_writes[0][2] == {"preserve_memory_snapshot": "true"}
    assert boundary.memory_writes[1][0] == "PUT"
    assert boundary.memory_writes[1][2] == {"preserve_memory_snapshot": "true"}

    # Verify persona was updated cleanly.
    assert len(boundary.persona_saves) == 1
    assert "Lumi" in boundary.persona.markdown
    assert "# Onboarding" not in boundary.persona.markdown

    # Verify cursor advanced AND existing core_memories_snapshot was NOT mutated.
    assert boundary.conversation.memory_extracted_through_message_id == messages[-1].id
    assert (
        boundary.conversation.core_memories_snapshot
        == "Core memories:\n- **Communication Style**: Casual tone"
    )


def test_memory_queue_skips_active_runs_and_uses_fallback_for_subscriptions(tmp_path):
    instructions = tmp_path / "AGENT.md"
    instructions.write_text("Base instructions.")
    base = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    messages = [
        Message(id=uuid4(), role="user", content="My name is Sam.", created_at=base)
    ]
    boundary = HybridBoundary(instructions, messages)
    boundary.active_runs = [boundary.make_run(messages[0].id)]
    fake_inf = SimpleNamespace(
        evaluate_decisions=AsyncMock(return_value={"should_save_memory": 0.0}),
        complete_text=AsyncMock(),
    )
    queue = MemoryQueue(boundary, agent_keys=lambda: ["alpha"], inference=fake_inf)

    # Skips while run is active.
    assert asyncio.run(queue.poll_once()) == 0
    fake_inf.evaluate_decisions.assert_not_awaited()

    # Once idle with subscription language provider (model_id=None), passes use_local_decisions=False.
    boundary.active_runs = []
    boundary.status = SimpleNamespace(state="ready", model_id=None, context_limit=32768)
    assert asyncio.run(queue.poll_once()) == 1
    assert fake_inf.evaluate_decisions.await_args.kwargs["use_local_decisions"] is False


def test_memory_queue_idle_debounce_waits_for_idle_delay_before_extracting(tmp_path):
    instructions = tmp_path / "AGENT.md"
    instructions.write_text("Base instructions.")
    base = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    messages = [
        Message(id=uuid4(), role="user", content="My name is Andrew.", created_at=base),
        Message(
            id=uuid4(),
            role="assistant",
            content="Nice to meet you!",
            created_at=base + timedelta(seconds=5),
        ),
    ]
    boundary = HybridBoundary(instructions, messages)
    fake_inf = SimpleNamespace(
        evaluate_decisions=AsyncMock(
            return_value={"should_save_memory": 0.05, "is_core_memory": 0.05}
        ),
        complete_text=AsyncMock(),
    )
    queue = MemoryQueue(
        boundary,
        agent_keys=lambda: ["alpha"],
        inference=fake_inf,
        idle_delay=MEMORY_QUEUE_IDLE_DELAY,
    )

    # 10 seconds after latest message (< 25s): skipped, schedules remaining ~15s wait.
    processed = asyncio.run(queue.poll_once(now=base + timedelta(seconds=15)))
    assert processed == 0
    fake_inf.evaluate_decisions.assert_not_awaited()
    assert queue._next_idle_wait == pytest.approx(15.0)
    assert boundary.conversation.memory_extracted_through_message_id is None

    # 25 seconds after latest message: processes immediately and advances cursor.
    processed_idle = asyncio.run(queue.poll_once(now=base + timedelta(seconds=30)))
    assert processed_idle == 1
    fake_inf.evaluate_decisions.assert_awaited_once()
    assert queue._next_idle_wait is None
    assert boundary.conversation.memory_extracted_through_message_id == messages[-1].id


def test_memory_queue_preempt_cancels_inflight_work_without_advancing_cursor(tmp_path):
    instructions = tmp_path / "AGENT.md"
    instructions.write_text("Base instructions.")
    base = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    messages = [
        Message(id=uuid4(), role="user", content="I love pour-over coffee.", created_at=base),
        Message(
            id=uuid4(),
            role="assistant",
            content="Great taste!",
            created_at=base + timedelta(seconds=2),
        ),
    ]
    boundary = HybridBoundary(instructions, messages)
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow_extract(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    fake_inf = SimpleNamespace(
        evaluate_decisions=AsyncMock(
            return_value={"should_save_memory": 0.95, "is_core_memory": 0.9}
        ),
        complete_text=AsyncMock(side_effect=slow_extract),
    )
    queue = MemoryQueue(
        boundary,
        agent_keys=lambda: ["alpha"],
        inference=fake_inf,
        idle_delay=timedelta(0),
    )

    async def run_scenario():
        poll_task = asyncio.create_task(queue.poll_once())
        await started.wait()
        assert queue.preempt() is True
        processed = await poll_task
        assert processed == 0
        assert cancelled.is_set()
        assert queue._active_work is None
        assert boundary.conversation.memory_extracted_through_message_id is None
        assert boundary.memory_writes == []

        # Retries cleanly on next quiet poll and advances cursor.
        fake_inf.complete_text = AsyncMock(
            return_value=json.dumps(
                {
                    "memories": [
                        {
                            "action": "create",
                            "memory_id": None,
                            "title": "Coffee",
                            "markdown": "Loves pour-over coffee.",
                            "tags": ["core"],
                        }
                    ],
                    "persona_markdown": None,
                }
            )
        )
        retried = await queue.poll_once()
        assert retried == 1
        assert boundary.conversation.memory_extracted_through_message_id == messages[-1].id
        assert len(boundary.memory_writes) == 1

    asyncio.run(run_scenario())


def test_agent_runner_reconcile_and_process_preempt_memory_queue():
    from kadan.api.pydantic_models.runners import Runner

    async def scenario():
        owner = uuid4()
        now = datetime.now(UTC)
        queued_run = AgentRun(
            id=uuid4(),
            agent_key="alpha",
            conversation_id=uuid4(),
            version=1,
            created_at=now,
            updated_at=now,
        )
        preempt_calls = []

        class StubServices:
            def __init__(self):
                self.memory_queue = SimpleNamespace(
                    preempt=lambda: preempt_calls.append("preempt") or True
                )
                self.runs = SimpleNamespace(owner=owner, process=self.process)

            async def process(self, run):
                await asyncio.sleep(0)

            async def json_request(self, service, method, path, **kwargs):
                if path == "/api/agent-runs":
                    return [queued_run.model_dump(mode="json")]
                return Runner(
                    runner_id="runner-1",
                    worker_id=owner,
                    capacity=5,
                    lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
                    agent_keys=["alpha"],
                ).model_dump(mode="json")

        services = StubServices()
        runner = AgentRunner(services, RunnerSettings(_env_file=None))
        try:
            await runner.reconcile()
            assert len(preempt_calls) == 1
        finally:
            await runner.close()

        # Also verify AgentRuns.process() calls preempt() upon claiming a run.
        claim_preempts = []
        claimed_run = queued_run.model_copy(
            update={
                "status": "running",
                "owner_id": owner,
                "lease_expires_at": datetime.now(UTC) + timedelta(seconds=60),
                "version": 2,
            }
        )
        proc_services = SimpleNamespace(
            client=None,
            settings=SimpleNamespace(hikari_url="http://unused"),
            memory_queue=SimpleNamespace(
                wakeup=asyncio.Event(),
                preempt=lambda: claim_preempts.append("claimed") or True,
            ),
            json_request=AsyncMock(return_value=claimed_run.model_dump(mode="json")),
        )
        runs = AgentRuns(proc_services, owner=owner)
        runs.advance = AsyncMock(return_value=None)
        await runs.process(queued_run)
        assert claim_preempts == ["claimed"]

    asyncio.run(scenario())
