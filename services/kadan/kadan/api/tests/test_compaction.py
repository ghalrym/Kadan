"""Unit tests for rolling conversation compaction, context assembly, and idle auto-compaction."""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from kadan.api.config import ServiceError
from kadan.api.pydantic_models.agent_runs import AgentRun, RunCall, RunMessage
from kadan.api.pydantic_models.resources import Message, Resource, ResourceKind
from kadan.runner.agent_inference import ToolCommand
from kadan.runner.agent_runs import AgentRuns
from kadan.runner.compaction import (
    COMPACTION_MAX_UNCOMPACTED_MESSAGES,
    IN_RUN_RECENT_TAIL_MESSAGES,
    IdleCompactor,
)
from kadan.runner.conversation_history import ReadHistoryArguments, history_page


class FakeInference:
    def __init__(self, *summaries):
        self.summaries = list(summaries)
        self.calls = []
        self.complete_calls = []

    async def complete_text(
        self,
        messages,
        model_id,
        conversation_key=None,
        *,
        priority="background",
        reuse_prefix=True,
    ):
        self.complete_calls.append(
            {
                "messages": messages,
                "model_id": model_id,
                "conversation_key": conversation_key,
                "priority": priority,
                "reuse_prefix": reuse_prefix,
            }
        )
        result = self.summaries.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    async def call(self, messages, model_id, conversation_key, tools, **kwargs):
        self.calls.append(messages)
        return RunCall(command=ToolCommand(tool="finish"))


class CompactionBoundary:
    def __init__(self, instruction_file, messages, **conv_overrides):
        now = datetime(2026, 11, 26, 15, 0, tzinfo=UTC)
        self.client = None
        self.settings = SimpleNamespace(
            hikari_url="http://unused", agent_instruction_file=str(instruction_file)
        )
        self.images = SimpleNamespace(wakeup=asyncio.Event())
        self.conversation = Resource(
            id=uuid4(),
            agent_key="alpha",
            kind=ResourceKind.CONVERSATION,
            title="New chat",
            messages=messages,
            version=len(messages) or 1,
            created_at=now,
            updated_at=now,
            **conv_overrides,
        )
        trigger = next(
            (item for item in reversed(messages) if item.role == "user"), None
        )
        self.current = AgentRun(
            id=uuid4(),
            agent_key="alpha",
            conversation_id=self.conversation.id,
            trigger_message_id=trigger.id if trigger else None,
            version=1,
            created_at=now,
            updated_at=now,
        )
        self.active_runs = []
        self.status = SimpleNamespace(
            state="ready", model_id="inkling-2-mini-iq4-nl", context_limit=32768
        )
        self.compaction_requests = []
        self.conflict_once = False
        self.conflict_always = False
        self.controller = AgentRuns(self)
        self.conversations = SimpleNamespace(all=self.list_conversations)

    async def list_conversations(self, key, kind):
        assert key == "alpha" and kind == ResourceKind.CONVERSATION
        return [self.conversation]

    async def resource(self, key, conversation_id):
        assert key == "alpha" and conversation_id == self.conversation.id
        return self.conversation

    async def agent(self, key):
        return SimpleNamespace(key=key, name="Alpha")

    async def agent_persona(self, key):
        return None

    async def inference_status(self):
        return self.status

    async def json_request(self, service, method, path, **kwargs):
        assert service == "Tsuchi"
        body = kwargs.get("json", {})
        if path == "/api/agent-runs" and method == "GET":
            return [run.model_dump(mode="json") for run in self.active_runs]
        if path == "/api/agents" and method == "GET":
            return [{"key": "alpha", "name": "Alpha"}]
        if (
            path == f"/api/agents/alpha/resources/{self.conversation.id}/runs"
            and method == "GET"
        ):
            return [self.current.model_dump(mode="json")]
        if (
            path == f"/api/agents/alpha/resources/{self.conversation.id}/compaction"
            and method == "PUT"
        ):
            self.compaction_requests.append(body)
            if self.conflict_always:
                raise ServiceError("This item changed.", 409, "version_conflict")
            if self.conflict_once:
                self.conflict_once = False
                # Simulate a follow-up user message arriving concurrently.
                self.conversation = self.conversation.model_copy(
                    update={
                        "messages": [
                            *self.conversation.messages,
                            Message(
                                role="user",
                                content="Follow-up while compacting",
                                created_at=datetime.now(UTC),
                            ),
                        ],
                        "version": self.conversation.version + 1,
                    }
                )
                raise ServiceError("This item changed.", 409, "version_conflict")
            if body["expected_version"] != self.conversation.version:
                raise ServiceError("This item changed.", 409, "version_conflict")
            if body.get("require_idle") and any(
                run.conversation_id == self.conversation.id
                and run.status in {"queued", "running"}
                for run in self.active_runs
            ):
                raise ServiceError("Conversation has active work", 409, "version_conflict")
            self.conversation = self.conversation.model_copy(
                update={
                    "compacted_summary": body["compacted_summary"],
                    "compacted_through_message_id": UUID(
                        body["compacted_through_message_id"]
                    ),
                    "compacted_at": datetime.fromisoformat(body["compacted_at"])
                    if body.get("compacted_at")
                    else datetime.now(UTC),
                    "version": self.conversation.version + 1,
                }
            )
            return self.conversation.model_dump(mode="json")
        raise AssertionError(f"Unexpected {method} {path}")


def make_messages(count: int, *, base: datetime, content_prefix: str = "Turn") -> list[Message]:
    return [
        Message(
            id=uuid4(),
            role="user" if index % 2 == 0 else "assistant",
            content=f"{content_prefix} {index}",
            created_at=base + timedelta(minutes=index),
        )
        for index in range(count)
    ]


def test_context_includes_compacted_summary_and_only_messages_after_anchor(tmp_path):
    instructions = tmp_path / "AGENT.md"
    instructions.write_text("Base instructions.")
    base = datetime(2026, 11, 26, 10, 0, tzinfo=UTC)
    messages = make_messages(5, base=base)
    boundary = CompactionBoundary(
        instructions,
        messages,
        compacted_summary="User is renovating a cedar greenhouse.",
        compacted_through_message_id=messages[2].id,
        compacted_at=base + timedelta(hours=1),
    )

    transcript = asyncio.run(boundary.controller.context(boundary.current, limit=20000))

    assert isinstance(transcript[0], RunMessage)
    assert transcript[0].role == "system"
    assert "Base instructions." in transcript[0].content
    assert "User is renovating a cedar greenhouse." in transcript[0].content
    # Only messages after anchor index 2 (i.e. indices 3 and 4) appear in prompt history.
    history_contents = [
        item.content
        for item in transcript[1:]
        if isinstance(item, RunMessage) and item.role in {"user", "assistant"}
    ]
    assert history_contents == ["Turn 3", "Turn 4"]
    # All original messages remain intact for history tools.
    page = history_page(boundary.conversation, ReadHistoryArguments(limit=10))
    assert [item["content"] for item in page["messages"]] == [
        f"Turn {index}" for index in range(5)
    ]


def test_in_run_budget_compaction_triggers_on_char_budget_and_preserves_tail(tmp_path):
    instructions = tmp_path / "AGENT.md"
    instructions.write_text("Base instructions.")
    base = datetime(2026, 11, 26, 10, 0, tzinfo=UTC)
    # 13 messages * ~1000 chars = 13,000 chars > 65% of 15,000 budget
    messages = [
        Message(
            id=uuid4(),
            role="user" if index % 2 == 0 else "assistant",
            content=f"Msg {index}: " + ("a" * 980),
            created_at=base + timedelta(minutes=index),
        )
        for index in range(13)
    ]
    boundary = CompactionBoundary(instructions, messages)
    fake = FakeInference("Rolling summary of messages 0..4.")
    boundary.controller.inference = fake

    transcript = asyncio.run(boundary.controller.context(boundary.current, limit=15000))

    assert len(fake.complete_calls) == 1
    assert fake.complete_calls[0]["priority"] == "interactive"
    assert fake.complete_calls[0]["conversation_key"] is None
    assert fake.complete_calls[0]["reuse_prefix"] is False
    assert len(boundary.compaction_requests) == 1
    req = boundary.compaction_requests[0]
    assert req["require_idle"] is False
    assert req["compacted_through_message_id"] == str(
        messages[-IN_RUN_RECENT_TAIL_MESSAGES - 1].id
    )
    assert boundary.conversation.compacted_summary == "Rolling summary of messages 0..4."
    # All 13 visible messages stay completely intact in storage.
    assert len(boundary.conversation.messages) == 13
    # Context includes rolling summary + the 8-message recent tail.
    assert "Rolling summary of messages 0..4." in transcript[0].content
    history_contents = [
        item.content
        for item in transcript[1:]
        if isinstance(item, RunMessage) and item.role in {"user", "assistant"}
    ]
    assert len(history_contents) == IN_RUN_RECENT_TAIL_MESSAGES
    assert history_contents[0].startswith("Msg 5:")
    assert history_contents[-1].startswith("Msg 12:")


def test_in_run_compaction_triggers_on_message_count_and_retries_version_conflict(
    tmp_path,
):
    instructions = tmp_path / "AGENT.md"
    instructions.write_text("Base instructions.")
    base = datetime(2026, 11, 26, 10, 0, tzinfo=UTC)
    total = COMPACTION_MAX_UNCOMPACTED_MESSAGES + 5
    messages = make_messages(total, base=base)
    boundary = CompactionBoundary(instructions, messages)
    boundary.conflict_once = True
    fake = FakeInference("Rolling summary across 60+ short turns.")
    boundary.controller.inference = fake

    transcript = asyncio.run(boundary.controller.context(boundary.current, limit=100000))

    # Retried cleanly after concurrent follow-up user message bumped version.
    assert len(boundary.compaction_requests) == 2
    assert (
        boundary.conversation.compacted_summary
        == "Rolling summary across 60+ short turns."
    )
    assert len(boundary.conversation.messages) == total + 1
    assert "Rolling summary across 60+ short turns." in transcript[0].content
    history_contents = [
        item.content
        for item in transcript[1:]
        if isinstance(item, RunMessage) and item.role in {"user", "assistant"}
    ]
    assert len(history_contents) == IN_RUN_RECENT_TAIL_MESSAGES
    assert history_contents[-1] == f"Turn {total - 1}"


def test_in_run_compaction_failure_falls_back_to_tail_history(tmp_path):
    instructions = tmp_path / "AGENT.md"
    instructions.write_text("Base instructions.")
    base = datetime(2026, 11, 26, 10, 0, tzinfo=UTC)
    messages = make_messages(COMPACTION_MAX_UNCOMPACTED_MESSAGES + 5, base=base)
    boundary = CompactionBoundary(instructions, messages)
    boundary.controller.inference = FakeInference(
        ServiceError("Temporary Hikari blip", 503)
    )

    transcript = asyncio.run(boundary.controller.context(boundary.current, limit=100000))

    assert boundary.compaction_requests == []
    assert boundary.conversation.compacted_summary is None
    assert transcript[-1].content == f"Turn {COMPACTION_MAX_UNCOMPACTED_MESSAGES + 4}"


def test_idle_compactor_skips_when_under_30_minutes(tmp_path):
    instructions = tmp_path / "AGENT.md"
    instructions.write_text("Base instructions.")
    base = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    messages = make_messages(4, base=base)  # latest at 12:03 UTC
    boundary = CompactionBoundary(instructions, messages)
    fake = FakeInference("Should not run")
    compactor = IdleCompactor(
        boundary, agent_keys=lambda: ["alpha"], inference=fake
    )

    count = asyncio.run(compactor.poll_once(now=base + timedelta(minutes=32)))

    assert count == 0
    assert fake.complete_calls == []
    assert boundary.compaction_requests == []


def test_idle_compactor_skips_when_run_is_active(tmp_path):
    instructions = tmp_path / "AGENT.md"
    instructions.write_text("Base instructions.")
    base = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    messages = make_messages(4, base=base)
    boundary = CompactionBoundary(instructions, messages)
    boundary.active_runs = [
        boundary.current.model_copy(
            update={
                "status": "running",
                "owner_id": uuid4(),
                "lease_expires_at": base + timedelta(hours=3),
            }
        )
    ]
    fake = FakeInference("Should not run")
    compactor = IdleCompactor(
        boundary, agent_keys=lambda: ["alpha"], inference=fake
    )

    count = asyncio.run(compactor.poll_once(now=base + timedelta(hours=2)))

    assert count == 0
    assert fake.complete_calls == []
    assert boundary.compaction_requests == []


def test_idle_compactor_skips_when_inference_not_ready(tmp_path):
    instructions = tmp_path / "AGENT.md"
    instructions.write_text("Base instructions.")
    base = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    messages = make_messages(4, base=base)
    boundary = CompactionBoundary(instructions, messages)
    boundary.status = SimpleNamespace(state="model_required", model_id=None)
    fake = FakeInference("Should not run")
    compactor = IdleCompactor(
        boundary, agent_keys=lambda: ["alpha"], inference=fake
    )

    count = asyncio.run(compactor.poll_once(now=base + timedelta(hours=2)))

    assert count == 0
    assert fake.complete_calls == []


def test_idle_compactor_compacts_after_30_minutes_with_background_priority(tmp_path):
    instructions = tmp_path / "AGENT.md"
    instructions.write_text("Base instructions.")
    base = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    messages = make_messages(4, base=base)  # latest at 12:03 UTC
    boundary = CompactionBoundary(
        instructions,
        messages,
        compacted_summary="Earlier summary.",
        compacted_through_message_id=messages[1].id,
        compacted_at=base + timedelta(minutes=1),
    )
    fake = FakeInference("Merged rolling summary through turn 3.")
    compactor = IdleCompactor(
        boundary, agent_keys=lambda: ["alpha"], inference=fake
    )

    count = asyncio.run(compactor.poll_once(now=base + timedelta(minutes=34)))

    assert count == 1
    assert len(fake.complete_calls) == 1
    assert fake.complete_calls[0]["priority"] == "background"
    assert fake.complete_calls[0]["conversation_key"] is None
    assert fake.complete_calls[0]["reuse_prefix"] is False
    prompt_text = fake.complete_calls[0]["messages"][1].content
    assert "Earlier summary." in prompt_text
    assert "Turn 2" in prompt_text and "Turn 3" in prompt_text
    assert "Turn 0" not in prompt_text
    assert len(boundary.compaction_requests) == 1
    assert boundary.compaction_requests[0]["require_idle"] is True
    assert boundary.compaction_requests[0]["compacted_through_message_id"] == str(
        messages[-1].id
    )
    assert boundary.conversation.compacted_summary == "Merged rolling summary through turn 3."
    assert len(boundary.conversation.messages) == 4

    # Polling again immediately does nothing because no uncompacted messages remain.
    second = asyncio.run(compactor.poll_once(now=base + timedelta(minutes=35)))
    assert second == 0
    assert len(fake.complete_calls) == 1


def test_idle_compactor_handles_concurrent_message_409_without_clobbering(tmp_path):
    instructions = tmp_path / "AGENT.md"
    instructions.write_text("Base instructions.")
    base = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    messages = make_messages(4, base=base)
    boundary = CompactionBoundary(instructions, messages)
    boundary.conflict_always = True
    fake = FakeInference("Summary raced with new message")
    compactor = IdleCompactor(
        boundary, agent_keys=lambda: ["alpha"], inference=fake
    )

    count = asyncio.run(compactor.poll_once(now=base + timedelta(hours=1)))

    assert count == 0
    assert boundary.conversation.compacted_summary is None
    assert len(boundary.compaction_requests) == 1
