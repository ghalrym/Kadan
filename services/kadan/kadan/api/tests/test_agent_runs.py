"""Agent orchestration tests with in-memory boundary fakes, never inference."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4, uuid5

import pytest
from kadan_tracing import carrier
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from kadan.api.config import ServiceError
from kadan.api.pydantic_models.agent_runs import AgentRun, ImageJob, RunMessage
from kadan.api.pydantic_models.computer import ComputerTask, ComputerTaskOrigin
from kadan.api.pydantic_models.resources import Message, Resource, ResourceKind
from kadan.runner.agent_inference import (
    REASONING_TOKENS,
    REPLY_TOKENS,
    ToolCommand,
    tool_characters,
)
from kadan.runner.agent_runs import (
    ARGUMENT_REJECTION,
    MAX_ARGUMENT_CORRECTIONS,
    MAX_CONTEXT,
    RESERVED_TOKENS,
    AgentRuns,
    RunState,
    context_characters,
)
from kadan.runner.tools import (
    ALL_TOOLS,
    CONVERSATION_TOOLS,
    MEMORY_TOOLS,
    PERSONA_TOOLS,
)


class Boundary:
    def __init__(self, instruction_file):
        now = datetime.now(UTC)
        self.client = None
        self.settings = SimpleNamespace(
            hikari_url="http://unused", agent_instruction_file=str(instruction_file)
        )
        self.images = SimpleNamespace(wakeup=asyncio.Event())
        self.computer = SimpleNamespace(wakeup=asyncio.Event())
        self.computer_tasks = {}
        self.sandbox = SimpleNamespace(configured=False)
        user = Message(role="user", content="Hello")
        self.conversation = Resource(
            id=uuid4(),
            agent_key="test",
            kind="conversation",
            title="Test",
            messages=[user],
            version=1,
            created_at=now,
            updated_at=now,
        )
        self.current = AgentRun(
            id=uuid4(),
            agent_key="test",
            conversation_id=self.conversation.id,
            trigger_message_id=user.id,
            version=1,
            created_at=now,
            updated_at=now,
        )
        self.jobs = {}
        self.deliveries = []
        self.delivery_attempts = []
        self.crash_after_delivery = False
        self.controller = AgentRuns(self)

    def start_owned(self, **changes):
        self.current = self.current.model_copy(
            update={
                "status": "running",
                "owner_id": self.controller.owner,
                "lease_expires_at": datetime.now(UTC) + timedelta(seconds=60),
                **changes,
            }
        )
        return self.current

    def cancel(self):
        self.current = self.current.model_copy(
            update={
                "status": "cancelled",
                "owner_id": None,
                "lease_expires_at": None,
                "version": self.current.version + 1,
            }
        )

    async def resource(self, key, conversation_id):
        assert key == "test" and conversation_id == self.conversation.id
        return self.conversation

    async def agent_persona(self, key):
        assert key == "test"

    async def inference_status(self):
        return SimpleNamespace(state="ready", model_id="selected-model")

    def check_owned(self, body):
        if self.current.status != "running" or body["worker_id"] != str(
            self.current.owner_id
        ):
            raise ServiceError("Run cancelled or lease lost", 409, "lease_lost")

    async def json_request(self, service, method, path, **kwargs):
        assert service == "Tsuchi"
        base = self.controller.path("test", self.current.id)
        body = kwargs.get("json", {})
        if path == base and method == "GET":
            return self.current.model_dump(mode="json")
        if path == base and method == "PUT":
            if self.current.status not in {"queued", "running"}:
                raise ServiceError("Terminal run cannot be claimed", 409)
            if body["expected_version"] != self.current.version:
                raise ServiceError("Stale run", 409)
            self.current = AgentRun.model_validate(
                {
                    **self.current.model_dump(),
                    **body,
                    "version": self.current.version + 1,
                }
            )
            return self.current.model_dump(mode="json")
        if path == base + "/messages" and method == "POST":
            self.check_owned(body)
            message = Message.model_validate(body["message"])
            self.delivery_attempts.append(message.id)
            if message.id not in self.current.delivered_message_ids:
                self.deliveries.append(message)
                self.current = self.current.model_copy(
                    update={
                        "delivered_message_ids": [
                            *self.current.delivered_message_ids,
                            message.id,
                        ],
                        "version": self.current.version + 1,
                    }
                )
                self.conversation = self.conversation.model_copy(
                    update={
                        "messages": [*self.conversation.messages, message],
                        "version": self.conversation.version + 1,
                    }
                )
            if self.crash_after_delivery:
                self.crash_after_delivery = False
                raise asyncio.CancelledError()
            return self.conversation.model_dump(mode="json")
        if path == base + "/image-jobs" and method == "POST":
            self.check_owned(body)
            job_id = UUID(body["id"])
            if job_id not in self.jobs:
                self.jobs[job_id] = ImageJob(
                    id=job_id,
                    agent_key="test",
                    run_id=self.current.id,
                    conversation_id=self.conversation.id,
                    prompt=body["prompt"],
                    request_id=UUID(body["request_id"]),
                    event_run_id=uuid5(job_id, "event"),
                    version=1,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            return self.jobs[job_id].model_dump(mode="json")
        if path.endswith("/image-jobs") and method == "GET":
            return [job.model_dump(mode="json") for job in self.jobs.values()]
        if "/image-jobs/" in path and method == "GET":
            return self.jobs[UUID(path.rsplit("/", 1)[1])].model_dump(mode="json")
        if path == "/api/agents/test/computer-tasks" and method == "GET":
            return [
                task.model_dump(mode="json")
                for task in self.computer_tasks.values()
                if task.status in {"queued", "running"}
            ]
        if path == "/api/agents/test/computer-tasks" and method == "POST":
            self.check_owned(body)
            task_id = UUID(body["id"])
            if task_id not in self.computer_tasks:
                self.computer_tasks[task_id] = ComputerTask(
                    id=task_id,
                    agent_key="test",
                    instructions=body["instructions"],
                    origin=ComputerTaskOrigin(
                        conversation_id=self.conversation.id,
                        run_id=UUID(body["origin_run_id"]),
                    ),
                    event_run_id=uuid5(task_id, "event"),
                    version=1,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            return self.computer_tasks[task_id].model_dump(mode="json")
        raise AssertionError(f"Unexpected {method} {path}")


def command(tool, **arguments):
    return ToolCommand(
        tool=tool, arguments=json.loads(json.dumps(arguments, default=str))
    )


def finish():
    return command("finish")


def send(**arguments):
    return command("send_message", **arguments)


def image_request(**arguments):
    return command("generate_image", **arguments)


class Commands:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []
        self.tools = []

    async def call(self, messages, model_id, conversation_key, tools):
        self.calls.append(messages)
        self.tools.append([tool.name for tool in tools])
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        if callable(result):
            result = result()
        return result


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    async def read_file(function):
        return function()

    # Thread-pool scheduling is an external boundary; these tests cover the read
    # and instruction lifetime without relying on executor-thread wakeups.
    monkeypatch.setattr("kadan.runner.agent_runs.asyncio.to_thread", read_file)
    instruction_file = tmp_path / "AGENT.md"
    instruction_file.write_text("Be clear and helpful.")
    return Boundary(instruction_file)


def test_finish_completes_without_delivering_a_message(boundary):
    boundary.controller.inference = Commands(finish())
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "completed"
    assert boundary.deliveries == []
    assert len(boundary.conversation.messages) == 1


def test_message_is_delivered_while_run_is_still_running(boundary):
    def finish_after_visible_delivery():
        assert boundary.current.status == "running"
        assert [item.content for item in boundary.deliveries] == ["Hello back"]
        return finish()

    boundary.controller.inference = Commands(
        send(end_run=False, content="Hello back"),
        finish_after_visible_delivery,
    )
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "completed"
    assert len(boundary.deliveries) == 1


def test_crash_after_delivery_replays_same_message_identity_once(boundary):
    boundary.crash_after_delivery = True
    boundary.controller.inference = Commands(send(end_run=True, content="Saved answer"))

    async def check():
        with pytest.raises(asyncio.CancelledError):
            await boundary.controller.advance(boundary.start_owned())
        assert boundary.current.transcript[-1].role == "assistant"
        boundary.controller.inference = Commands()
        await boundary.controller.advance(boundary.current)

    asyncio.run(check())
    assert len(boundary.deliveries) == 1
    assert boundary.delivery_attempts == [boundary.deliveries[0].id] * 2
    assert boundary.current.status == "completed"


def test_image_submission_returns_queued_job_without_waiting_or_delivery(boundary):
    boundary.controller.inference = Commands(
        image_request(prompt="A little blue bird"),
    )
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "completed"
    assert len(boundary.jobs) == 1
    assert len(boundary.controller.inference.calls) == 1
    image = next(iter(boundary.jobs.values()))
    assert image.status == "queued"
    assert boundary.images.wakeup.is_set()
    assert boundary.deliveries == []
    result = json.loads(
        boundary.current.transcript[-1].content.removeprefix("Tool result: ")
    )
    assert result == {
        "tool": "generate_image",
        "job_id": str(image.id),
        "status": "queued",
        "delivered": False,
    }


@pytest.mark.parametrize(
    "status,delivered,error",
    [
        ("completed", False, None),
        ("completed", True, None),
        ("failed", False, "Image worker unavailable"),
        ("cancelled", False, None),
    ],
)
def test_image_event_context_carries_result_and_delivery_state(
    boundary, status, delivered, error
):
    job_id = uuid4()
    image = ImageJob(
        id=job_id,
        agent_key="test",
        run_id=uuid4(),
        conversation_id=boundary.conversation.id,
        prompt="Blue bird",
        request_id=uuid4(),
        event_run_id=boundary.current.id,
        status=status,
        asset_id=uuid4() if status == "completed" else None,
        delivered_message_id=uuid4() if delivered else None,
        error=error,
        version=1,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    boundary.jobs[job_id] = image
    run = boundary.current.model_copy(
        update={"trigger_job_id": job_id, "trigger_message_id": None}
    )
    context = asyncio.run(boundary.controller.context(run))
    event = json.loads(context[-1].content.removeprefix("Image job event: "))
    assert event == {
        "job_id": str(job_id),
        "status": status,
        "prompt": "Blue bird",
        "error": error,
        "delivered": delivered,
    }


def test_failure_after_visible_message_does_not_append_an_answer(boundary):
    boundary.controller.inference = Commands(
        send(end_run=False, content="Already delivered"),
        ServiceError("Hikari failed", 503),
    )
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "failed"
    assert boundary.current.error == "Hikari failed"
    assert [item.content for item in boundary.deliveries] == ["Already delivered"]


def test_cancellation_during_inference_prevents_its_message_delivery(boundary):
    def cancelled_result():
        boundary.cancel()
        return send(end_run=False, content="Should not be sent")

    boundary.controller.inference = Commands(cancelled_result)
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "cancelled"
    assert boundary.deliveries == []


def test_cancelled_run_cannot_execute_persisted_command(boundary):
    run = boundary.start_owned(
        transcript=[
            RunMessage(
                role="assistant",
                content=send(end_run=False, content="No").model_dump_json(),
            )
        ]
    )
    boundary.cancel()
    boundary.controller.inference = Commands()
    with pytest.raises(ServiceError) as error:
        asyncio.run(boundary.controller.advance(run))
    assert error.value.code == "lease_lost"
    assert boundary.deliveries == []
    assert boundary.controller.inference.calls == []


def test_process_cannot_reclaim_a_cancelled_run(boundary):
    boundary.cancel()
    boundary.controller.inference = Commands()
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "cancelled"
    assert boundary.controller.inference.calls == []


def test_instructions_are_reread_when_a_new_context_is_built(boundary):
    first = asyncio.run(boundary.controller.context(boundary.current))
    with open(boundary.settings.agent_instruction_file, "w") as file:
        file.write("Use brief messages.")
    second = asyncio.run(
        boundary.controller.context(boundary.current.model_copy(update={"id": uuid4()}))
    )
    assert first[0].content.startswith("Be clear and helpful.")
    assert second[0].content.startswith("Use brief messages.")


def test_invalid_instruction_encoding_fails_before_inference(boundary):
    instruction_file = Path(boundary.settings.agent_instruction_file)
    instruction_file.write_bytes(b"\xff")
    boundary.controller.inference = Commands()
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "failed"
    assert str(instruction_file) in boundary.current.error
    assert "could not be decoded" in boundary.current.error
    assert "text encoding" in boundary.current.error
    assert boundary.current.owner_id is None
    assert boundary.current.lease_expires_at is None
    assert boundary.current.transcript == []
    assert boundary.current.delivered_message_ids == []
    assert boundary.controller.inference.calls == []
    assert boundary.delivery_attempts == []
    assert boundary.deliveries == []


@pytest.mark.parametrize("failure", ["missing", "directory"])
def test_unreadable_instructions_fail_before_inference(boundary, failure):
    instruction_file = Path(boundary.settings.agent_instruction_file)
    instruction_file.unlink()
    if failure == "directory":
        instruction_file.mkdir()
    boundary.controller.inference = Commands()
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "failed"
    assert str(instruction_file) in boundary.current.error
    assert boundary.current.owner_id is None
    assert boundary.current.lease_expires_at is None
    assert boundary.current.transcript == []
    assert boundary.controller.inference.calls == []
    assert boundary.delivery_attempts == []
    assert boundary.deliveries == []


@pytest.mark.parametrize("failure", ["malformed", "missing"])
def test_recovery_uses_persisted_instructions_without_reloading_file(boundary, failure):
    transcript = [
        RunMessage(role="system", content="Saved instructions"),
        RunMessage(role="user", content="Existing context"),
    ]
    boundary.current = boundary.current.model_copy(update={"transcript": transcript})
    instruction_file = Path(boundary.settings.agent_instruction_file)
    if failure == "malformed":
        instruction_file.write_bytes(b"\xff")
    else:
        instruction_file.unlink()
    boundary.controller.inference = Commands(finish())
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "completed"
    assert boundary.current.owner_id is None
    assert boundary.current.lease_expires_at is None
    assert boundary.current.transcript[:2] == transcript
    assert len(boundary.controller.inference.calls) == 1
    assert [item.content for item in boundary.controller.inference.calls[0]] == [
        item.content for item in transcript
    ]
    assert boundary.deliveries == []


def test_repaired_instructions_allow_a_new_queued_run(boundary):
    queued = boundary.current
    instruction_file = Path(boundary.settings.agent_instruction_file)
    instruction_file.write_bytes(b"\xff")
    boundary.controller.inference = Commands(finish())
    asyncio.run(boundary.controller.process(boundary.current))
    failed = boundary.current
    assert failed.status == "failed"
    assert boundary.controller.inference.calls == []
    boundary.current = queued.model_copy(update={"id": uuid4()})
    instruction_file.write_text("Repaired instructions.")
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "completed"
    assert boundary.current.owner_id is None
    assert boundary.current.lease_expires_at is None
    assert boundary.current.error is None
    assert boundary.current.transcript[0].content == "Repaired instructions."
    assert len(boundary.controller.inference.calls) == 1
    assert boundary.controller.inference.calls[0][0].content == "Repaired instructions."
    assert boundary.deliveries == []


@pytest.mark.parametrize("loss", ["cancelled", "another_owner", "expired"])
def test_instruction_decode_failure_respects_lease_fencing(boundary, monkeypatch, loss):
    Path(boundary.settings.agent_instruction_file).write_bytes(b"\xff")
    boundary.controller.inference = Commands()

    async def read_after_ownership_loss(function):
        if loss == "cancelled":
            boundary.cancel()
        else:
            boundary.current = boundary.current.model_copy(
                update={"owner_id": uuid4()}
                if loss == "another_owner"
                else {"lease_expires_at": datetime.now(UTC) - timedelta(seconds=1)}
            )
        return function()

    monkeypatch.setattr(
        "kadan.runner.agent_runs.asyncio.to_thread", read_after_ownership_loss
    )
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == (
        "cancelled" if loss == "cancelled" else "running"
    )
    assert boundary.current.error is None
    if loss == "cancelled":
        assert boundary.current.owner_id is None
        assert boundary.current.lease_expires_at is None
    elif loss == "another_owner":
        assert boundary.current.owner_id != boundary.controller.owner
    else:
        assert boundary.current.lease_expires_at < datetime.now(UTC)
    assert boundary.controller.inference.calls == []
    assert boundary.delivery_attempts == []
    assert boundary.deliveries == []


def test_recovery_can_replay_the_last_allowed_command(boundary):
    transcript = []
    for _ in range(23):
        transcript.extend(
            [
                RunMessage(
                    role="assistant",
                    content='{"tool":"send_message","content":"Earlier"}',
                ),
                RunMessage(role="user", content="Tool result: delivered"),
            ]
        )
    transcript.append(RunMessage(role="assistant", content='{"tool":"finish"}'))
    boundary.controller.inference = Commands()
    asyncio.run(
        boundary.controller.advance(boundary.start_owned(transcript=transcript))
    )
    assert boundary.current.status == "completed"
    assert boundary.controller.inference.calls == []


def test_lease_renewal_waits_for_delivery_before_reading_run_version(boundary):
    async def check():
        delivery_started = asyncio.Event()
        release_delivery = asyncio.Event()
        original = boundary.json_request
        reads_during_delivery = []

        async def delayed_request(service, method, path, **kwargs):
            if method == "POST" and path.endswith("/messages"):
                delivery_started.set()
                await release_delivery.wait()
            if (
                method == "GET"
                and delivery_started.is_set()
                and not release_delivery.is_set()
            ):
                reads_during_delivery.append(path)
            return await original(service, method, path, **kwargs)

        boundary.json_request = delayed_request
        boundary.controller.inference = Commands(
            send(end_run=False, content="Delivered once"),
            finish(),
        )
        run = boundary.start_owned()
        advance = asyncio.create_task(boundary.controller.advance(run))
        await delivery_started.wait()
        # This is the state update heartbeat performs after its timer fires.
        renewal = asyncio.create_task(
            boundary.controller.update(
                run,
                lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
            )
        )
        await asyncio.sleep(0)
        assert not renewal.done()
        assert reads_during_delivery == []
        release_delivery.set()
        await asyncio.gather(renewal, advance)

    asyncio.run(check())
    assert boundary.current.status == "completed"
    assert len(boundary.deliveries) == 1


def test_final_message_releases_ownership_without_completion_inference(boundary):
    boundary.controller.inference = Commands(send(content="Done", end_run=True))
    asyncio.run(boundary.controller.process(boundary.current))
    assert len(boundary.controller.inference.calls) == 1
    assert [message.content for message in boundary.deliveries] == ["Done"]
    assert boundary.current.status == "completed"
    assert boundary.current.owner_id is None
    assert boundary.current.lease_expires_at is None
    assert boundary.current.transcript[-1].content.startswith("Tool result:")


def test_crash_after_image_submission_replays_job_without_inference(boundary):
    original = boundary.json_request
    interrupted = False

    async def crash_after_job(service, method, path, **kwargs):
        nonlocal interrupted
        result = await original(service, method, path, **kwargs)
        if method == "POST" and path.endswith("/image-jobs") and not interrupted:
            interrupted = True
            raise asyncio.CancelledError()
        return result

    boundary.json_request = crash_after_job
    boundary.controller.inference = Commands(image_request(prompt="A blue bird"))

    async def check():
        with pytest.raises(asyncio.CancelledError):
            await boundary.controller.advance(boundary.start_owned())
        assert len(boundary.jobs) == 1
        assert boundary.current.transcript[-1].role == "assistant"
        boundary.controller.inference = Commands()
        await boundary.controller.advance(boundary.current)
        assert boundary.controller.inference.calls == []

    asyncio.run(check())
    assert boundary.current.status == "completed"
    assert len(boundary.jobs) == 1
    assert boundary.images.wakeup.is_set()


def test_next_context_sees_pending_image_after_submitting_run_finishes(boundary):
    boundary.controller.inference = Commands(image_request(prompt="A blue bird"))
    asyncio.run(boundary.controller.process(boundary.current))
    context = asyncio.run(boundary.controller.context(boundary.current))
    image = next(iter(boundary.jobs.values()))
    assert str(image.id) in context[-1].content
    assert '"status": "queued"' in context[-1].content
    assert '"prompt": "A blue bird"' in context[-1].content


def test_pending_job_context_cannot_exceed_run_budget(boundary, monkeypatch):
    boundary.controller.inference = Commands(image_request(prompt="A blue bird"))
    asyncio.run(boundary.controller.process(boundary.current))
    jobs = boundary.jobs
    boundary.jobs = {}
    baseline = asyncio.run(boundary.controller.context(boundary.current))
    budget = (
        sum(len(message.content) for message in baseline)
        + tool_characters(boundary.controller.bound_tools(boundary.current))
        + 10
    )
    monkeypatch.setattr("kadan.runner.agent_runs.MAX_CONTEXT", budget)
    assert asyncio.run(boundary.controller.context(boundary.current)) == baseline
    boundary.jobs = jobs
    with pytest.raises(ServiceError, match="context limit"):
        asyncio.run(boundary.controller.context(boundary.current))


def test_history_budget_reserves_bound_tool_definitions(boundary, monkeypatch):
    baseline = asyncio.run(boundary.controller.context(boundary.current))
    budget = sum(len(message.content) for message in baseline)
    monkeypatch.setattr("kadan.runner.agent_runs.MAX_CONTEXT", budget)
    with pytest.raises(ServiceError, match="context"):
        asyncio.run(boundary.controller.context(boundary.current))


def test_saved_context_reserves_bound_tool_definitions_before_inference(
    boundary, monkeypatch
):
    transcript = [RunMessage(role="user", content="Existing context")]
    monkeypatch.setattr(
        "kadan.runner.agent_runs.MAX_CONTEXT", len(transcript[0].content)
    )
    boundary.controller.inference = Commands(finish())
    run = boundary.start_owned(transcript=transcript)
    with pytest.raises(ServiceError, match="context limit"):
        asyncio.run(boundary.controller.advance(run))
    assert boundary.controller.inference.calls == []


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_image_event_resumes_graph_in_same_conversation(boundary, status):
    job_id = uuid4()
    boundary.jobs[job_id] = ImageJob(
        id=job_id,
        agent_key="test",
        run_id=uuid4(),
        conversation_id=boundary.conversation.id,
        prompt="Blue bird",
        request_id=uuid4(),
        event_run_id=boundary.current.id,
        status=status,
        asset_id=uuid4() if status == "completed" else None,
        error="Worker failed" if status == "failed" else None,
        version=1,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    boundary.current = boundary.current.model_copy(
        update={"trigger_job_id": job_id, "trigger_message_id": None}
    )
    boundary.controller.inference = Commands(
        send(
            content="Here it is" if status == "completed" else "Image failed",
            image_job_id=job_id if status == "completed" else None,
            end_run=True,
        )
    )
    receipts = []
    original = boundary.json_request

    async def record_delivery(service, method, path, **kwargs):
        if method == "POST" and path.endswith("/messages"):
            receipts.append(kwargs["json"])
        return await original(service, method, path, **kwargs)

    boundary.json_request = record_delivery
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "completed"
    assert boundary.current.conversation_id == boundary.conversation.id
    assert len(receipts) == len(boundary.deliveries) == 1
    assert receipts[0]["image_job_id"] == (
        str(job_id) if status == "completed" else None
    )
    context = boundary.controller.inference.calls[0]
    assert f'"status": "{status}"' in context[-1].content
    assert len(boundary.controller.inference.calls) == 1


def test_invalid_commands_fail_the_run_without_a_retry(boundary):
    boundary.controller.inference = Commands(
        ServiceError("Invalid command", 502, "invalid_command"), finish()
    )
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "failed"
    assert len(boundary.controller.inference.calls) == 1
    assert boundary.deliveries == []


def test_cancelling_graph_task_cancels_inflight_inference(boundary):
    async def check():
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def infer(*args):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        boundary.controller.inference = SimpleNamespace(call=infer)
        task = asyncio.create_task(boundary.controller.advance(boundary.start_owned()))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()

    asyncio.run(check())
    assert boundary.deliveries == []
    assert boundary.current.transcript[-1].role == "user"


def test_graph_allows_full_command_budget(boundary):
    boundary.controller.inference = Commands(
        *[send(content=f"Message {index}", end_run=False) for index in range(23)],
        finish(),
    )
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "completed"
    assert len(boundary.deliveries) == 23
    assert len(boundary.controller.inference.calls) == 24


def test_runner_restores_durable_trace_parent(boundary, monkeypatch):

    provider = TracerProvider(shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("unit")
    with trace.use_span(tracer.start_span("api.enqueue"), end_on_exit=True) as parent:
        parent_id = parent.get_span_context().span_id
        boundary.current = boundary.current.model_copy(
            update={"trace_parent": carrier()["traceparent"]}
        )
    monkeypatch.setattr(trace, "get_tracer", lambda *args, **kwargs: tracer)
    boundary.controller.inference = Commands(finish())
    asyncio.run(boundary.controller.process(boundary.current))
    execution = next(
        span
        for span in exporter.get_finished_spans()
        if span.name == "kadan.agent_runs.process"
    )
    assert execution.parent.span_id == parent_id
    assert boundary.current.status == "completed"
    assert not any(
        "prompt" in key or "content" in key
        for span in exporter.get_finished_spans()
        for key in span.attributes
    )
    provider.shutdown()


class Computer:
    """The sandbox client boundary; records calls instead of reaching a computer."""

    configured = True

    def __init__(self, error=None):
        self.calls = []
        self.error = error

    async def run(self, method, path, **request):
        self.calls.append((method, path, request))
        if self.error:
            raise self.error
        return {"status": 200, "body": {"output": "hi"}}


def tool_result(boundary, index=-1):
    return json.loads(
        boundary.current.transcript[index].content.removeprefix("Tool result: ")
    )


def test_every_conversation_and_computer_tool_is_bound(boundary):
    boundary.controller.inference = Commands(finish())
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.controller.inference.tools == [
        [
            tool.name
            for tool in [*CONVERSATION_TOOLS, *MEMORY_TOOLS, *PERSONA_TOOLS]
            if tool.name != "read_request_references"
        ]
    ]

    boundary.sandbox = Computer()
    boundary.current = boundary.current.model_copy(
        update={"status": "queued", "transcript": []}
    )
    boundary.controller.inference = Commands(finish())
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.controller.inference.tools == [
        [tool.name for tool in ALL_TOOLS if tool.name != "read_request_references"]
    ]


def test_use_computer_hands_over_tells_the_user_and_ends_the_run(boundary):
    boundary.sandbox = Computer()
    boundary.controller.inference = Commands(
        command("use_computer", instructions="Open the docs", message="On it!")
    )
    asyncio.run(boundary.controller.process(boundary.current))
    [task] = boundary.computer_tasks.values()
    assert task.instructions == "Open the docs"
    assert task.origin.run_id == boundary.current.id
    assert boundary.computer.wakeup.is_set()
    assert [message.content for message in boundary.deliveries] == ["On it!"]
    assert boundary.current.status == "completed"
    assert boundary.sandbox.calls == []
    assert tool_result(boundary)["task_id"] == str(task.id)


def test_pending_computer_tasks_are_in_the_context(boundary):
    boundary.sandbox = Computer()
    boundary.controller.inference = Commands(
        command("use_computer", instructions="Open the docs", message="On it!")
    )
    asyncio.run(boundary.controller.process(boundary.current))
    context = asyncio.run(boundary.controller.context(boundary.current))
    assert "Pending computer tasks (do not resubmit)" in context[-1].content
    assert "Open the docs" in context[-1].content


def test_computer_tools_are_not_available_to_the_conversation_agent(boundary):
    boundary.sandbox = Computer()
    boundary.controller.inference = Commands(command("bash_exec", command="ls"))
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "failed"
    assert "unknown tool" in boundary.current.error
    assert boundary.sandbox.calls == []


@pytest.mark.parametrize("loss", ["cancelled", "another_owner", "expired"])
def test_ownership_lost_after_command_saved_prevents_computer_dispatch(boundary, loss):
    boundary.sandbox = Computer()
    boundary.controller.inference = Commands(
        command("use_computer", instructions="Run a command", message="On it")
    )

    async def check():
        state = await boundary.controller.infer_command(
            RunState(
                run=boundary.start_owned(
                    transcript=[RunMessage(role="user", content="Run a command")]
                )
            )
        )
        assert (
            boundary.current.transcript[-1].content
            == state["command"].model_dump_json()
        )
        if loss == "cancelled":
            boundary.cancel()
        else:
            changes = (
                {"owner_id": uuid4()}
                if loss == "another_owner"
                else {"lease_expires_at": datetime.now(UTC) - timedelta(seconds=1)}
            )
            boundary.current = boundary.current.model_copy(update=changes)
        with pytest.raises(ServiceError, match="no longer owned"):
            await boundary.controller.invoke_tool(RunState(**state))

    asyncio.run(check())
    assert boundary.computer_tasks == {}
    assert boundary.current.transcript[-1].role == "assistant"


@pytest.mark.parametrize("parent_id", [None, uuid4()])
def test_workspace_creation_uses_the_run_agent_without_a_computer(boundary, parent_id):
    document_id = uuid4()
    saved = boundary.conversation.model_copy(
        update={
            "id": document_id,
            "kind": ResourceKind.SPACE,
            "title": "Notes",
            "markdown": "# Plan",
            "messages": [],
            "parent_id": parent_id,
        }
    )
    original_request = boundary.json_request

    async def request(service, method, path, **kwargs):
        if method == "POST":
            assert (service, path, kwargs) == (
                "Tsuchi",
                "/api/agents/test/resources",
                {
                    "json": {
                        "kind": "space",
                        "title": "Notes",
                        "markdown": "# Plan",
                        "parent_id": str(parent_id) if parent_id else None,
                    }
                },
            )
            return saved.model_dump(mode="json")
        return await original_request(service, method, path, **kwargs)

    boundary.json_request = request
    run = boundary.start_owned()
    asyncio.run(
        boundary.controller.invoke_tool(
            RunState(
                run=run,
                command=command(
                    "create_workspace_document",
                    title=" Notes ",
                    markdown="# Plan",
                    parent_id=parent_id,
                ),
            )
        )
    )
    assert tool_result(boundary) == {
        "tool": "create_workspace_document",
        "document_id": str(document_id),
        "title": "Notes",
        "parent_id": str(parent_id) if parent_id else None,
        "version": 1,
    }
    assert boundary.current.status == "running"
    assert boundary.deliveries == []


@pytest.mark.parametrize("status", [400, 404, 409, 503])
def test_workspace_write_failure_is_reported_without_success_or_retry(boundary, status):
    original_request = boundary.json_request
    writes = []

    async def request(service, method, path, **kwargs):
        if method == "POST":
            writes.append(path)
            raise ServiceError(
                "Parent unavailable" if status < 500 else "Timed out", status
            )
        return await original_request(service, method, path, **kwargs)

    boundary.json_request = request
    asyncio.run(
        boundary.controller.invoke_tool(
            RunState(
                run=boundary.start_owned(),
                command=command(
                    "create_workspace_document", title="Notes", parent_id=uuid4()
                ),
            )
        )
    )
    result = tool_result(boundary)
    assert "document_id" not in result
    assert ("may have been saved" in result["error"]) is (status >= 500)
    assert len(writes) == 1


def test_restart_does_not_repeat_an_unconfirmed_workspace_write(boundary):
    original_request = boundary.json_request

    async def request(service, method, path, **kwargs):
        assert method != "POST"
        return await original_request(service, method, path, **kwargs)

    boundary.json_request = request
    asyncio.run(
        boundary.controller.invoke_tool(
            RunState(
                run=boundary.start_owned(),
                command=command("create_workspace_document", title="Notes"),
                replayed=True,
            )
        )
    )
    assert "check Workspace before retrying" in tool_result(boundary)["error"]
    assert "document_id" not in tool_result(boundary)


def test_cancelled_run_cannot_create_a_workspace_document(boundary):
    run = boundary.start_owned()
    boundary.cancel()
    with pytest.raises(ServiceError, match="no longer owned"):
        asyncio.run(
            boundary.controller.invoke_tool(
                RunState(
                    run=run, command=command("create_workspace_document", title="Notes")
                )
            )
        )


def test_invalid_workspace_response_is_an_uncertain_write(boundary):
    original_request = boundary.json_request

    async def request(service, method, path, **kwargs):
        if method == "POST":
            return {"id": str(uuid4())}
        return await original_request(service, method, path, **kwargs)

    boundary.json_request = request
    asyncio.run(
        boundary.controller.invoke_tool(
            RunState(
                run=boundary.start_owned(),
                command=command("create_workspace_document", title="Notes"),
            )
        )
    )
    assert "may have been saved" in tool_result(boundary)["error"]
    assert "document_id" not in tool_result(boundary)


@pytest.mark.parametrize(
    "invalid",
    [
        command("shell_exec_command", command="ls"),
        command("missing_tool"),
        command("browser_navigate", address="https://example.com"),
    ],
)
def test_unknown_or_disabled_tools_fail_without_correction(boundary, invalid):
    boundary.controller.inference = Commands(invalid, finish())
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "failed"
    assert len(boundary.controller.inference.calls) == 1
    assert boundary.deliveries == []


@pytest.mark.parametrize(
    "invalid, field, code",
    [
        (
            command("read_conversation_history", unsupported="private"),
            "unsupported",
            "extra_forbidden",
        ),
        (command("read_conversation_history", limit=21), "limit", "less_than_equal"),
        (
            command("read_conversation_history", start_message_id="private"),
            "start_message_id",
            "uuid_parsing",
        ),
        (command("search_conversation_history"), "query", "missing"),
    ],
)
def test_history_argument_rejection_is_useful_and_never_executes(
    boundary, monkeypatch, invalid, field, code
):
    from unittest.mock import AsyncMock

    from kadan.runner.agent_runs import RunConversation

    history = AsyncMock(side_effect=AssertionError("Rejected call executed"))
    monkeypatch.setattr(RunConversation, "conversation_history", history)
    boundary.controller.inference = Commands(invalid, finish())
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "completed"
    assert len(boundary.controller.inference.calls) == 2
    feedback = boundary.controller.inference.calls[1][-1]
    assert feedback.type == "human"
    rejection = json.loads(feedback.content.removeprefix(ARGUMENT_REJECTION))
    assert rejection["tool"] == invalid.tool
    assert rejection["problems"] == [{"field": field, "code": code}]
    assert "No tool was executed" in rejection["error"]
    assert "private" not in json.dumps(boundary.current.model_dump(mode="json"))
    assert invalid.model_dump_json() not in [
        item.content for item in boundary.current.transcript
    ]
    history.assert_not_awaited()
    assert boundary.deliveries == []


def test_corrected_history_call_returns_result_then_delivers_once(boundary):
    boundary.controller.inference = Commands(
        command("read_conversation_history", unsupported="private"),
        command("read_conversation_history", limit=1),
        send(content="Found the history", end_run=True),
    )
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "completed"
    assert len(boundary.controller.inference.calls) == 3
    result = json.loads(boundary.controller.inference.calls[2][-1].content)
    assert result["messages"][0]["content"] == "Hello"
    assert [message.content for message in boundary.deliveries] == ["Found the history"]


def test_argument_correction_exhaustion_retains_safe_diagnostics(boundary):
    invalid = command("read_conversation_history", unsupported="private")
    boundary.controller.inference = Commands(*([invalid] * 4))
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "failed"
    assert len(boundary.controller.inference.calls) == MAX_ARGUMENT_CORRECTIONS + 1
    assert "correction attempts" in boundary.current.error
    assert "read_conversation_history" in boundary.current.error
    assert "unsupported: extra_forbidden" in boundary.current.error
    assert (
        len(
            [
                item
                for item in boundary.current.transcript
                if item.content.startswith(ARGUMENT_REJECTION)
            ]
        )
        == 3
    )
    assert "private" not in json.dumps(boundary.current.model_dump(mode="json"))
    assert boundary.deliveries == []


def test_successful_tool_does_not_reset_argument_correction_budget(boundary):
    invalid = command("read_conversation_history", unsupported=True)
    boundary.controller.inference = Commands(
        invalid,
        command("read_conversation_history", limit=1),
        invalid,
        invalid,
        finish(),
    )
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "failed"
    assert len(boundary.controller.inference.calls) == 4
    assert "correction attempts" in boundary.current.error


def test_argument_rejection_consumes_remaining_command_budget(boundary):
    boundary.controller.inference = Commands(
        *[send(content="Progress", end_run=False) for _ in range(23)],
        command("read_conversation_history", unsupported=True),
        finish(),
    )
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "failed"
    assert len(boundary.controller.inference.calls) == 24
    assert "24-command run limit" in boundary.current.error
    assert len(boundary.deliveries) == 23


def test_argument_diagnostics_are_bounded_and_redact_unconventional_keys(boundary):
    boundary.controller.inference = Commands(
        command(
            "read_conversation_history",
            **{f"private content {index}": "secret" for index in range(50)},
        ),
        finish(),
    )
    asyncio.run(boundary.controller.process(boundary.current))
    feedback = boundary.controller.inference.calls[1][-1].content
    rejection = json.loads(feedback.removeprefix(ARGUMENT_REJECTION))
    assert len(rejection["problems"]) == 5
    assert all(problem["field"] == "[redacted]" for problem in rejection["problems"])
    assert len(feedback) < 2000
    assert "private content" not in feedback and "secret" not in feedback


def test_invalid_mutations_have_no_effect_before_correction(boundary):
    boundary.sandbox = Computer()
    boundary.controller.inference = Commands(
        send(content="Must not deliver"),
        command(
            "use_computer", instructions="Do it", message="On it", unsupported=True
        ),
        finish(),
    )
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "completed"
    assert boundary.deliveries == []
    assert boundary.computer_tasks == {}


@pytest.mark.parametrize("loss", ["cancelled", "owner", "expired"])
def test_ownership_loss_during_invalid_inference_cannot_save_or_correct(boundary, loss):
    def rejected_after_loss():
        if loss == "cancelled":
            boundary.cancel()
        else:
            boundary.current = boundary.current.model_copy(
                update={
                    "owner_id": uuid4()
                    if loss == "owner"
                    else boundary.controller.owner,
                    "lease_expires_at": datetime.now(UTC) - timedelta(seconds=1)
                    if loss == "expired"
                    else boundary.current.lease_expires_at,
                }
            )
        return command("read_conversation_history", unsupported=True)

    boundary.controller.inference = Commands(rejected_after_loss, finish())
    asyncio.run(boundary.controller.process(boundary.current))
    assert len(boundary.controller.inference.calls) == 1
    assert not any(
        item.content.startswith(ARGUMENT_REJECTION)
        for item in boundary.current.transcript
    )
    assert boundary.deliveries == []


@pytest.mark.parametrize("exhausted", [False, True])
def test_restart_preserves_argument_feedback_and_correction_limit(boundary, exhausted):
    async def check():
        run = boundary.start_owned()
        run = (await boundary.controller.prepare(RunState(run=run)))["run"]
        boundary.controller.inference = Commands(
            *([command("read_conversation_history", unsupported="private")] * 3)
        )
        for _ in range(3 if exhausted else 1):
            result = await boundary.controller.infer_command(RunState(run=run))
            run = result["run"]
            assert result["command"] is None
        saved = run.transcript[-1].content
        boundary.controller = AgentRuns(boundary)
        boundary.controller.inference = Commands(
            send(content="Recovered", end_run=True)
        )
        await boundary.controller.process(boundary.current)
        if exhausted:
            assert boundary.current.status == "failed"
            assert boundary.controller.inference.calls == []
            assert boundary.deliveries == []
        else:
            assert boundary.current.status == "completed"
            assert boundary.controller.inference.calls[0][-1].content == saved
            assert len(boundary.deliveries) == 1

    asyncio.run(check())


def test_cancelled_saved_rejection_cannot_resume_correction(boundary):
    async def check():
        boundary.controller.inference = Commands(
            command("read_conversation_history", unsupported=True)
        )
        run = (await boundary.controller.prepare(RunState(run=boundary.start_owned())))[
            "run"
        ]
        await boundary.controller.infer_command(RunState(run=run))
        boundary.cancel()
        boundary.controller.inference = Commands(finish())
        await boundary.controller.process(boundary.current)

    asyncio.run(check())
    assert boundary.current.status == "cancelled"
    assert boundary.controller.inference.calls == []
    assert boundary.deliveries == []


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (None, MAX_CONTEXT),
        (1, 0),
        (1024, 0),
        (4096, 0),
        (RESERVED_TOKENS, 0),
        (RESERVED_TOKENS + 1, 3),
        (RESERVED_TOKENS + 4096, 4096 * 3),
        (65536, (65536 - REASONING_TOKENS - REPLY_TOKENS - 1024) * 3),
    ],
)
def test_transcript_budget_follows_the_models_context_window(tokens, expected):
    assert context_characters(tokens) == expected


def test_transcript_budget_leaves_room_for_the_bound_tools(boundary):
    status = SimpleNamespace(context_limit=65536)
    assert boundary.controller.transcript_limit(
        status, boundary.current
    ) == context_characters(65536) - tool_characters(
        [
            tool
            for tool in [*CONVERSATION_TOOLS, *MEMORY_TOOLS, *PERSONA_TOOLS]
            if tool.name != "read_request_references"
        ]
    )
    boundary.controller.services.sandbox.configured = True
    assert boundary.controller.transcript_limit(
        status, boundary.current
    ) == context_characters(65536) - tool_characters(
        [tool for tool in ALL_TOOLS if tool.name != "read_request_references"]
    )


def test_smaller_model_window_drops_history_before_inference(boundary):
    # Leave room for bound tool schemas while keeping the older turn too large.
    async def inference_status():
        return SimpleNamespace(
            state="ready",
            model_id="selected-model",
            context_limit=RESERVED_TOKENS + 7168,
        )

    boundary.inference_status = inference_status
    older = Message(role="assistant", content="x" * 30000)
    boundary.conversation = boundary.conversation.model_copy(
        update={"messages": [older, *boundary.conversation.messages]}
    )
    boundary.controller.inference = Commands(finish())
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "completed"
    assert len(boundary.controller.inference.calls) == 1
    assert not any(
        str(older.id) in item.content for item in boundary.current.transcript
    )


def test_smaller_model_window_rejects_an_oversized_saved_transcript(boundary):
    async def inference_status():
        return SimpleNamespace(
            state="ready",
            model_id="selected-model",
            context_limit=RESERVED_TOKENS + 3072,
        )

    boundary.inference_status = inference_status
    boundary.current = boundary.current.model_copy(
        update={"transcript": [RunMessage(role="user", content="x" * 10000)]}
    )
    boundary.controller.inference = Commands(finish())
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "failed"
    assert "context limit" in boundary.current.error
    assert boundary.controller.inference.calls == []


@pytest.mark.parametrize(
    "context_limit", [1, 1024, 1025, 4096, RESERVED_TOKENS, RESERVED_TOKENS + 1]
)
@pytest.mark.parametrize("saved_transcript", [False, True])
def test_window_without_room_for_tools_fails_before_inference(
    boundary, context_limit, saved_transcript
):
    async def inference_status():
        return SimpleNamespace(
            state="ready", model_id="selected-model", context_limit=context_limit
        )

    boundary.inference_status = inference_status
    if saved_transcript:
        boundary.current = boundary.current.model_copy(
            update={"transcript": [RunMessage(role="user", content="Hello")]}
        )
    boundary.controller.inference = Commands(finish())
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "failed"
    assert "context" in boundary.current.error
    assert boundary.controller.inference.calls == []
    assert boundary.current.owner_id is None
    assert boundary.current.lease_expires_at is None


def test_zero_context_budget_does_not_fall_back_to_default(boundary):
    with pytest.raises(ServiceError, match="context limit"):
        asyncio.run(boundary.controller.context(boundary.current, limit=0))


@pytest.mark.parametrize("loss", ["message_limit", "budget", "removed", "role_changed"])
def test_quoted_trigger_id_does_not_admit_an_absent_user_message(boundary, loss):
    target = boundary.conversation.messages[0]
    quote = Message(role="user", content=f"The earlier message ID was {target.id}")
    if loss == "message_limit":
        messages = [
            target,
            *[Message(role="user", content="Later") for _ in range(96)],
            quote,
        ]
    elif loss == "budget":
        messages = [target, Message(role="assistant", content="x" * MAX_CONTEXT), quote]
    elif loss == "removed":
        messages = [quote]
    else:
        messages = [target.model_copy(update={"role": "assistant"}), quote]
    boundary.conversation = boundary.conversation.model_copy(
        update={"messages": messages}
    )
    boundary.controller.inference = Commands(finish())
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "failed"
    assert (
        boundary.current.error
        == "The triggering message is outside the agent context window."
    )
    assert boundary.current.owner_id is None
    assert boundary.current.lease_expires_at is None
    assert boundary.controller.inference.calls == []
    assert boundary.deliveries == []


def test_actual_user_trigger_is_admitted_at_the_existing_history_limit(boundary):
    target = boundary.conversation.messages[0]
    messages = [
        Message(role="assistant", content="Older excluded history"),
        target,
        *[Message(role="user", content=f"Later {index}") for index in range(96)],
    ]
    boundary.conversation = boundary.conversation.model_copy(
        update={"messages": messages}
    )
    boundary.controller.inference = Commands(finish())
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "completed"
    assert len(boundary.controller.inference.calls) == 1
    admitted = boundary.controller.inference.calls[0][1:-1]
    assert [item.content for item in admitted] == [
        f"[message {message.id}] {message.content}" for message in messages[-97:]
    ]


def test_context_without_a_user_trigger_keeps_its_existing_admission(boundary):
    boundary.current = boundary.current.model_copy(update={"trigger_message_id": None})
    boundary.controller.inference = Commands(finish())
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "completed"
    assert len(boundary.controller.inference.calls) == 1


def test_runs_keep_history_and_tool_results_up_to_the_models_window(boundary):
    async def inference_status():
        return SimpleNamespace(
            state="ready", model_id="selected-model", context_limit=65536
        )

    boundary.inference_status = inference_status
    long = Message(role="assistant", content="x" * 60000)
    boundary.conversation = boundary.conversation.model_copy(
        update={"messages": [long, *boundary.conversation.messages]}
    )
    boundary.controller.inference = Commands(finish())
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "completed"
    assert any("x" * 60000 in item.content for item in boundary.current.transcript)


def test_no_tool_call_completes_after_one_inference_without_delivery(boundary):
    boundary.controller.inference = Commands(finish())
    asyncio.run(boundary.controller.process(boundary.current))
    assert len(boundary.controller.inference.calls) == 1
    assert boundary.current.status == "completed"
    assert boundary.current.owner_id is None
    assert boundary.current.lease_expires_at is None
    assert boundary.deliveries == []


def persona_record(boundary, markdown="Use concise answers."):
    return boundary.conversation.model_copy(
        update={
            "id": uuid4(),
            "kind": ResourceKind.PERSONA,
            "markdown": markdown,
            "messages": [],
        }
    )


def test_owned_persona_applies_without_a_selected_persona(boundary, monkeypatch):
    persona = persona_record(boundary)

    async def read(key):
        assert key == "test"
        return persona

    monkeypatch.setattr(boundary, "agent_persona", read)
    first = asyncio.run(boundary.controller.context(boundary.current))
    assert "Your saved persona:\nUse concise answers." in first[0].content
    persona.markdown = "Use detailed answers."
    second = asyncio.run(boundary.controller.context(boundary.current))
    assert "Use detailed answers." in second[0].content
    assert "Use concise answers." in first[0].content


@pytest.mark.parametrize("same_persona", [True, False])
def test_explicit_persona_supplements_owned_persona_once(
    boundary, monkeypatch, same_persona
):
    owned = persona_record(boundary, "Owned instructions")
    selected = (
        owned if same_persona else persona_record(boundary, "Selected instructions")
    )
    original_resource = boundary.resource

    async def read(key):
        return owned

    async def resource(key, resource_id):
        if resource_id == selected.id:
            return selected
        return await original_resource(key, resource_id)

    monkeypatch.setattr(boundary, "agent_persona", read)
    monkeypatch.setattr(boundary, "resource", resource)
    run = boundary.current.model_copy(update={"persona_id": selected.id})
    context = asyncio.run(boundary.controller.context(run))
    assert context[0].content.count("Owned instructions") == 1
    if not same_persona:
        assert context[0].content.endswith("Selected persona:\nSelected instructions")


def test_recovery_does_not_reread_owned_persona(boundary, monkeypatch):
    async def unavailable(key):
        raise AssertionError("Prepared work must keep its persona snapshot")

    monkeypatch.setattr(boundary, "agent_persona", unavailable)
    boundary.controller.inference = Commands(finish())
    run = boundary.start_owned(
        transcript=[RunMessage(role="system", content="Saved persona")]
    )
    asyncio.run(boundary.controller.advance(run))
    assert boundary.current.status == "completed"
    assert boundary.current.transcript[0].content == "Saved persona"


def test_persona_save_uses_agent_scope_and_read_version(boundary, monkeypatch):
    persona = persona_record(boundary)
    original_request = boundary.json_request
    writes = []

    async def request(service, method, path, **kwargs):
        if path.endswith("/persona"):
            writes.append((method, path, kwargs["json"]))
            return persona.model_dump(mode="json")
        return await original_request(service, method, path, **kwargs)

    monkeypatch.setattr(boundary, "json_request", request)
    boundary.controller.inference = Commands(
        command(
            "update_persona",
            title="My persona",
            markdown="Be concise.",
            expected_version=None,
        ),
        finish(),
    )
    asyncio.run(boundary.controller.process(boundary.current))
    assert writes == [
        (
            "PUT",
            "/api/agents/test/persona",
            {
                "title": "My persona",
                "markdown": "Be concise.",
                "expected_version": None,
            },
        )
    ]
    assert tool_result(boundary, -2)["persona"]["version"] == 1
    assert boundary.current.status == "completed"


def test_recovered_persona_write_is_not_repeated(boundary):
    pending = command(
        "update_persona", title="My persona", markdown="Be concise.", expected_version=1
    )
    run = boundary.start_owned(
        transcript=[
            RunMessage(role="system", content="Saved instructions"),
            RunMessage(role="assistant", content=pending.model_dump_json()),
        ]
    )
    boundary.controller.inference = Commands(finish())
    asyncio.run(boundary.controller.advance(run))
    assert "read_persona before retrying" in tool_result(boundary, -2)["error"]
    assert len(boundary.controller.inference.calls) == 1


def test_recovered_persona_read_is_repeated(boundary):
    run = boundary.start_owned(
        transcript=[
            RunMessage(role="system", content="Saved instructions"),
            RunMessage(
                role="assistant", content=command("read_persona").model_dump_json()
            ),
        ]
    )
    boundary.controller.inference = Commands(finish())
    asyncio.run(boundary.controller.advance(run))
    assert tool_result(boundary, -2) == {"persona": None}


def test_cancelled_run_cannot_start_persona_save(boundary):
    from kadan.runner.agent_runs import RunConversation
    from kadan.runner.tools.persona import UpdatePersonaArguments

    conversation = RunConversation(boundary.controller, boundary.start_owned())
    boundary.cancel()
    with pytest.raises(ServiceError, match="no longer owned"):
        asyncio.run(
            conversation.update_persona(
                UpdatePersonaArguments(
                    title="Persona",
                    markdown="Be concise.",
                    expected_version=None,
                )
            )
        )


def test_persona_conflict_is_a_tool_result(boundary, monkeypatch):
    original_request = boundary.json_request

    async def conflict(service, method, path, **kwargs):
        if path.endswith("/persona"):
            raise ServiceError("Persona changed; read it again.", 409)
        return await original_request(service, method, path, **kwargs)

    monkeypatch.setattr(boundary, "json_request", conflict)
    boundary.controller.inference = Commands(
        command(
            "update_persona",
            title="Persona",
            markdown="Be concise.",
            expected_version=2,
        ),
        finish(),
    )
    asyncio.run(boundary.controller.process(boundary.current))
    assert tool_result(boundary, -2) == {"error": "Persona changed; read it again."}
    assert boundary.current.status == "completed"


def selected_reference(markdown="Saved selected text"):
    from kadan.api.pydantic_models.agent_runs import ReferenceSnapshot

    return ReferenceSnapshot(
        resource_id=uuid4(), kind="space", title="Plan", version=3, markdown=markdown
    )


@pytest.mark.parametrize("computer", [False, True])
@pytest.mark.parametrize("selected", [False, True])
def test_reference_scope_matches_budget_preparation_binding_and_validation(
    boundary, computer, selected
):
    boundary.sandbox.configured = computer
    boundary.current.references = [selected_reference()] if selected else []
    expected = (
        ALL_TOOLS if computer else [*CONVERSATION_TOOLS, *MEMORY_TOOLS, *PERSONA_TOOLS]
    )
    expected = [
        tool for tool in expected if tool.name != "read_request_references" or selected
    ]
    assert boundary.controller.bound_tools(boundary.current) == expected
    status = SimpleNamespace(context_limit=65536)
    assert boundary.controller.transcript_limit(status, boundary.current) == (
        context_characters(65536) - tool_characters(expected)
    )
    boundary.controller.inference = Commands(
        command("read_request_references"), finish()
    )
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.controller.inference.tools[0] == [tool.name for tool in expected]
    if selected:
        assert boundary.current.status == "completed"
        assert (
            tool_result(boundary, -2)["references"][0]["markdown"]
            == "Saved selected text"
        )
        assert all(
            "Saved selected text" not in item.content
            for item in boundary.current.transcript
            if item.role == "system" or not item.content.startswith("Tool result: ")
        )
    else:
        assert boundary.current.status == "failed"
        assert "unknown tool" in boundary.current.error
        assert not any(
            item.content.startswith("Tool result: ")
            for item in boundary.current.transcript
        )


def test_saved_reference_reader_replays_from_global_registry_without_selection(
    boundary,
):
    boundary.start_owned(
        transcript=[
            RunMessage(role="system", content="Saved instructions"),
            RunMessage(
                role="assistant",
                content=command("read_request_references").model_dump_json(),
            ),
        ]
    )
    boundary.controller.inference = Commands(finish())
    asyncio.run(boundary.controller.advance(boundary.current))
    assert tool_result(boundary, -2)["references"] == []
    assert boundary.current.status == "completed"
    assert "read_request_references" not in boundary.controller.inference.tools[0]


def test_cancelled_run_cannot_replay_saved_reference_reader(boundary):
    boundary.start_owned(
        references=[selected_reference()],
        transcript=[
            RunMessage(
                role="assistant",
                content=command("read_request_references").model_dump_json(),
            )
        ],
    )
    boundary.cancel()
    with pytest.raises(ServiceError, match="no longer owned"):
        asyncio.run(boundary.controller.infer_command(RunState(run=boundary.current)))
    assert len(boundary.current.transcript) == 1


def test_selected_reference_over_context_budget_fails_without_truncation(boundary):
    boundary.current.references = [selected_reference("x" * 12000)]
    context = asyncio.run(boundary.controller.context(boundary.current))
    reserve = tool_characters(boundary.controller.bound_tools(boundary.current))
    tokens = (
        RESERVED_TOKENS
        + (reserve + sum(len(item.content) for item in context) + 1000) // 3
    )

    async def status():
        return SimpleNamespace(
            state="ready", model_id="selected-model", context_limit=tokens
        )

    boundary.inference_status = status
    boundary.controller.inference = Commands(
        command("read_request_references"), finish()
    )
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "failed"
    assert "context limit" in boundary.current.error
    assert len(boundary.controller.inference.calls) == 1
    assert tool_result(boundary)["references"][0]["markdown"] == "x" * 12000


def codex_start_command(**changes):
    return command(
        "codex_start",
        **{
            "prompt": "Build a hello world server",
            "cwd": "/home/gem/hello",
            "message": "Codex is building it now.",
            **changes,
        },
    )


@pytest.fixture
def codex_starts(boundary, monkeypatch):
    from kadan.api.pydantic_models.coding import CodexStarted

    boundary.sandbox.configured = True
    starts = []

    async def start(self, task, request_id, origin):
        starts.append((task, request_id, origin))
        if task.cwd == "/rejected":
            raise ServiceError("Use an absolute project folder.", 409, "codex_rejected")
        return CodexStarted(thread_id=uuid4(), cwd=task.cwd, running=True)

    monkeypatch.setattr("kadan.runner.agent_runs.CodingClient.start", start)
    return starts


def test_codex_start_tells_the_user_and_ends_the_response(boundary, codex_starts):
    boundary.controller.inference = Commands(codex_start_command())
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "completed"
    assert [item.content for item in boundary.deliveries] == [
        "Codex is building it now."
    ]
    assert len(boundary.controller.inference.calls) == 1
    task, request_id, origin = codex_starts[0]
    assert task.cwd == "/home/gem/hello"
    assert request_id == boundary.deliveries[0].id
    assert (origin.agent_key, origin.conversation_id, origin.run_id) == (
        "test",
        boundary.conversation.id,
        boundary.current.id,
    )


def test_codex_start_failure_returns_to_the_model_without_a_message(
    boundary, codex_starts
):
    boundary.controller.inference = Commands(
        codex_start_command(cwd="/rejected"),
        send(content="Codex could not start.", end_run=True),
    )
    asyncio.run(boundary.controller.process(boundary.current))
    assert tool_result(boundary, -3)["error"] == "Use an absolute project folder."
    assert [item.content for item in boundary.deliveries] == ["Codex could not start."]


def test_codex_event_run_starts_from_the_event(boundary):
    from kadan.api.pydantic_models.agent_runs import RunEvent

    run = boundary.current.model_copy(
        update={
            "trigger_message_id": None,
            "trigger_event": RunEvent(kind="codex", content='{"status": "completed"}'),
        }
    )
    context = asyncio.run(boundary.controller.context(run))
    assert context[-1].content == 'Codex thread event: {"status": "completed"}'
