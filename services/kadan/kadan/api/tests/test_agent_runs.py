"""Agent orchestration tests with in-memory boundary fakes, never inference."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4, uuid5

import pytest

from kadan.api.agent_inference import Finish, GenerateImage, SendMessage
from kadan.api.agent_runs import AgentRuns
from kadan.api.config import ServiceError
from kadan.api.pydantic_models.agent_runs import AgentRun, ImageJob, RunMessage
from kadan.api.pydantic_models.resources import Message, Resource


class Boundary:
    def __init__(self, instruction_file):
        now = datetime.now(UTC)
        self.client = None
        self.settings = SimpleNamespace(
            hikari_url="http://unused", agent_instruction_file=str(instruction_file)
        )
        self.images = SimpleNamespace(wakeup=asyncio.Event())
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
        raise AssertionError(f"Unexpected {method} {path}")


class Commands:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    async def call(self, messages, model_id, conversation_key):
        self.calls.append(messages)
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
    monkeypatch.setattr("kadan.api.agent_runs.asyncio.to_thread", read_file)
    instruction_file = tmp_path / "AGENT.md"
    instruction_file.write_text("Be clear and helpful.")
    return Boundary(instruction_file)


def test_finish_completes_without_delivering_a_message(boundary):
    boundary.controller.inference = Commands(Finish(tool="finish"))
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "completed"
    assert boundary.deliveries == []
    assert len(boundary.conversation.messages) == 1


def test_message_is_delivered_while_run_is_still_running(boundary):
    def finish_after_visible_delivery():
        assert boundary.current.status == "running"
        assert [item.content for item in boundary.deliveries] == ["Hello back"]
        return Finish(tool="finish")

    boundary.controller.inference = Commands(
        SendMessage(tool="send_message", end_run=False, content="Hello back"),
        finish_after_visible_delivery,
    )
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "completed"
    assert len(boundary.deliveries) == 1


def test_crash_after_delivery_replays_same_message_identity_once(boundary):
    boundary.crash_after_delivery = True
    boundary.controller.inference = Commands(
        SendMessage(tool="send_message", end_run=True, content="Saved answer")
    )

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
        GenerateImage(tool="generate_image", prompt="A little blue bird"),
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
        SendMessage(tool="send_message", end_run=False, content="Already delivered"),
        ServiceError("Hikari failed", 503),
    )
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "failed"
    assert boundary.current.error == "Hikari failed"
    assert [item.content for item in boundary.deliveries] == ["Already delivered"]


def test_cancellation_during_inference_prevents_its_message_delivery(boundary):
    def cancelled_result():
        boundary.cancel()
        return SendMessage(
            tool="send_message", end_run=False, content="Should not be sent"
        )

    boundary.controller.inference = Commands(cancelled_result)
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "cancelled"
    assert boundary.deliveries == []


def test_cancelled_run_cannot_execute_persisted_command(boundary):
    run = boundary.start_owned(
        transcript=[
            RunMessage(
                role="assistant",
                content=SendMessage(
                    tool="send_message", end_run=False, content="No"
                ).model_dump_json(),
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


def test_recovery_uses_persisted_instructions_without_reloading_file(boundary):
    boundary.controller.inference = Commands(Finish(tool="finish"))
    run = boundary.start_owned(
        transcript=[RunMessage(role="user", content="Existing context")]
    )
    boundary.settings.agent_instruction_file = "/does-not-exist/AGENT.md"
    asyncio.run(boundary.controller.advance(run))
    assert boundary.current.status == "completed"


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
            SendMessage(tool="send_message", end_run=False, content="Delivered once"),
            Finish(tool="finish"),
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
    boundary.controller.inference = Commands(
        SendMessage(tool="send_message", content="Done", end_run=True)
    )
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
    boundary.controller.inference = Commands(
        GenerateImage(tool="generate_image", prompt="A blue bird")
    )

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
    boundary.controller.inference = Commands(
        GenerateImage(tool="generate_image", prompt="A blue bird")
    )
    asyncio.run(boundary.controller.process(boundary.current))
    context = asyncio.run(boundary.controller.context(boundary.current))
    image = next(iter(boundary.jobs.values()))
    assert str(image.id) in context[-1].content
    assert '"status": "queued"' in context[-1].content
    assert '"prompt": "A blue bird"' in context[-1].content


def test_pending_job_context_cannot_exceed_run_budget(boundary, monkeypatch):
    boundary.controller.inference = Commands(
        GenerateImage(tool="generate_image", prompt="A blue bird")
    )
    asyncio.run(boundary.controller.process(boundary.current))
    jobs = boundary.jobs
    boundary.jobs = {}
    baseline = asyncio.run(boundary.controller.context(boundary.current))
    budget = sum(len(message.content) for message in baseline) + 10
    monkeypatch.setattr("kadan.api.agent_runs.MAX_CONTEXT", budget)
    assert asyncio.run(boundary.controller.context(boundary.current)) == baseline
    boundary.jobs = jobs
    with pytest.raises(ServiceError, match="context limit"):
        asyncio.run(boundary.controller.context(boundary.current))


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
        SendMessage(
            tool="send_message",
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


@pytest.mark.parametrize("invalid_attempts", [2, 3])
def test_graph_retains_invalid_command_correction_limit(boundary, invalid_attempts):
    boundary.controller.inference = Commands(
        *[
            ServiceError("Invalid command", 502, "invalid_command")
            for _ in range(invalid_attempts)
        ],
        Finish(tool="finish"),
    )
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == (
        "completed" if invalid_attempts == 2 else "failed"
    )
    assert len(boundary.controller.inference.calls) == 3
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


def test_graph_allows_full_command_budget_and_corrections(boundary):
    boundary.controller.inference = Commands(
        ServiceError("Invalid command", 502, "invalid_command"),
        ServiceError("Invalid command", 502, "invalid_command"),
        *[
            SendMessage(tool="send_message", content=f"Message {index}", end_run=False)
            for index in range(23)
        ],
        Finish(tool="finish"),
    )
    asyncio.run(boundary.controller.process(boundary.current))
    assert boundary.current.status == "completed"
    assert len(boundary.deliveries) == 23
    assert len(boundary.controller.inference.calls) == 26
