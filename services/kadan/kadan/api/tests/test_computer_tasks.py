"""The computer agent's worker, with in-memory storage, computer and inference."""

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from kadan.api.config import ServiceError
from kadan.api.pydantic_models.agent_runs import RunMessage, ToolCommand
from kadan.api.pydantic_models.computer import (
    ComputerSteering,
    ComputerTask,
    ComputerTaskOrigin,
    ComputerThread,
)
from kadan.runner.computer_tasks import (
    EARLIER,
    NEW_STEERING,
    STEERING,
    TASK_END,
    TASK_START,
    ComputerTasks,
    compact,
)


class Storage:
    """Tsuchi's computer task and thread routes, in memory."""

    def __init__(self, tmp_path):
        now = datetime.now(UTC)
        instructions = tmp_path / "COMPUTER_AGENT.md"
        instructions.write_text("Operate the computer.")
        self.client = None
        self.settings = SimpleNamespace(
            hikari_url="http://unused", computer_instruction_file=str(instructions)
        )
        self.sandbox = Computer()
        self.thread = ComputerThread(agent_key="test", version=1, updated_at=now)
        self.tasks: dict[UUID, ComputerTask] = {}
        self.steering: list[ComputerSteering] = []
        self.arrive_before_finish: list[str] = []
        self.events = []

    def add_task(self, instructions="Open the docs", origin=True):
        now = datetime.now(UTC)
        task = ComputerTask(
            id=uuid4(),
            agent_key="test",
            instructions=instructions,
            origin=ComputerTaskOrigin(conversation_id=uuid4(), run_id=uuid4())
            if origin
            else None,
            version=1,
            created_at=now,
            updated_at=now,
        )
        self.tasks[task.id] = task
        return task

    def steer(self, content):
        self.steering.append(
            ComputerSteering(
                id=uuid4(),
                task_id=next(
                    task.id
                    for task in self.tasks.values()
                    if task.status in {"queued", "running"}
                ),
                content=content,
                created_at=datetime.now(UTC),
            )
        )

    def current(self):
        return self.thread.model_copy(update={"steering": list(self.steering)})

    async def inference_status(self):
        return SimpleNamespace(state="ready", model_id="model", context_limit=None)

    async def json_request(self, service, method, path, json=None, **kwargs):
        assert service == "Tsuchi"
        active = [
            task.model_dump(mode="json")
            for task in self.tasks.values()
            if task.status in {"queued", "running"}
        ]
        if path in {"/api/computer-tasks", "/api/agents/test/computer-tasks"}:
            return active
        if path.startswith("/api/agents/test/computer-tasks/"):
            return self.tasks[UUID(path.rsplit("/", 1)[1])].model_dump(mode="json")
        assert path == "/api/agents/test/computer-thread"
        if method == "GET":
            return self.current().model_dump(mode="json")
        return self.put(json)

    def put(self, body):
        if body["expected_version"] != self.thread.version:
            raise ServiceError("This item changed.", 409)
        consumed = set(body["consumed_steering_ids"])
        task = body["task"]
        if task and task["status"] in {"completed", "failed"}:
            for content in self.arrive_before_finish:
                self.steer(content)
            self.arrive_before_finish = []
            if {str(item.id) for item in self.steering} - consumed:
                raise ServiceError(NEW_STEERING, 409)
        self.steering = [item for item in self.steering if str(item.id) not in consumed]
        if task:
            current = self.tasks[UUID(task["id"])]
            if current.status != "cancelled":
                self.tasks[current.id] = current.model_copy(
                    update={"status": task["status"], "summary": task.get("summary")}
                )
                if task["status"] in {"completed", "failed"}:
                    self.events.append(self.tasks[current.id])
        self.thread = ComputerThread.model_validate(
            {
                **self.thread.model_dump(),
                "transcript": body["transcript"],
                "owner_id": body["owner_id"],
                "lease_expires_at": body["lease_expires_at"],
                "version": self.thread.version + 1,
            }
        )
        return self.current().model_dump(mode="json")


class Computer:
    configured = True

    def __init__(self, error=None):
        self.calls = []
        self.error = error

    async def run(self, method, path, **request):
        self.calls.append((method, path, request))
        if self.error:
            raise self.error
        return {"status": 200, "body": {"output": "hi"}}


class Commands:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    async def call(self, messages, model_id, conversation_key, tools):
        self.calls.append(messages)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def command(tool, **arguments):
    return ToolCommand(tool=tool, arguments=arguments)


def finish(summary="Done.", **arguments):
    return command("finish_task", summary=summary, **arguments)


@pytest.fixture
def storage(tmp_path):
    return Storage(tmp_path)


def serve(storage, *commands):
    tasks = ComputerTasks(storage)
    tasks.inference = Commands(*commands)
    asyncio.run(tasks.serve("test"))
    return tasks.inference


def contents(storage):
    return [item.content for item in storage.thread.transcript]


def test_task_runs_computer_tools_then_reports_its_summary(storage):
    task = storage.add_task()
    serve(storage, command("bash_exec", command="echo hi"), finish("Printed hi."))
    method, path, request = storage.sandbox.calls[0]
    assert (method, path) == ("POST", "/v1/bash/exec")
    assert request["json_body"]["command"] == "echo hi"
    done = storage.tasks[task.id]
    assert (done.status, done.summary) == ("completed", "Printed hi.")
    assert storage.events == [done]
    assert contents(storage)[0].startswith(TASK_START)
    assert json.loads(contents(storage)[-1].removeprefix(TASK_END))["status"] == (
        "completed"
    )
    assert storage.thread.owner_id is None


def test_unsuccessful_finish_fails_the_task(storage):
    task = storage.add_task()
    serve(storage, finish("No such site.", succeeded=False))
    assert storage.tasks[task.id].status == "failed"


def test_user_steering_is_read_before_the_next_inference(storage):
    storage.add_task(origin=False)
    storage.steer("Use the terminal instead")
    calls = serve(storage, finish())
    assert calls.calls[0][-1].content == STEERING + "Use the terminal instead"
    assert storage.steering == []


def test_steering_that_arrives_before_finishing_keeps_the_task_going(storage):
    task = storage.add_task()
    storage.arrive_before_finish = ["Also open the changelog"]
    calls = serve(storage, finish("Docs open."), finish("Docs and changelog open."))
    assert len(calls.calls) == 2
    second = [message.content for message in calls.calls[1]]
    assert any("new steering" in item for item in second)
    assert second[-1] == STEERING + "Also open the changelog"
    assert storage.tasks[task.id].summary == "Docs and changelog open."


def test_thread_keeps_earlier_tasks_for_later_ones(storage):
    first = storage.add_task("Open the docs")
    serve(storage, finish("Docs open."))
    storage.add_task("Close the docs")
    calls = serve(storage, finish())
    seen = " ".join(message.content for message in calls.calls[0])
    assert str(first.id) in seen and "Docs open." in seen and "Close the docs" in seen


def test_tasks_run_one_at_a_time_in_order(storage):
    first, second = storage.add_task("First"), storage.add_task("Second")
    calls = serve(storage, finish("One."), finish("Two."))
    assert "First" in calls.calls[0][-1].content
    assert "Second" in calls.calls[1][-1].content
    assert storage.events == [storage.tasks[first.id], storage.tasks[second.id]]


def test_sandbox_errors_become_tool_results(storage):
    storage.sandbox = Computer(ServiceError("The computer is unavailable"))
    storage.add_task()
    calls = serve(storage, command("bash_exec", command="ls"), finish())
    assert "The computer is unavailable" in calls.calls[1][-1].content


@pytest.mark.parametrize(
    ("tool", "repeated"), [("bash_exec", False), ("browser_get_markdown", True)]
)
def test_restart_never_repeats_an_unconfirmed_side_effect(storage, tool, repeated):
    task = storage.add_task()
    task = storage.tasks[task.id] = task.model_copy(update={"status": "running"})
    start = {"task_id": str(task.id), "from": "agent", "instructions": "Do it"}
    pending = command(tool, **({"command": "ls"} if tool == "bash_exec" else {}))
    storage.thread.transcript = [
        RunMessage(role="user", content=TASK_START + json.dumps(start)),
        RunMessage(role="assistant", content=pending.model_dump_json()),
    ]
    storage.steer("Use a different approach")
    calls = serve(storage, finish())
    assert bool(storage.sandbox.calls) is repeated
    messages = [item.content for item in calls.calls[0]]
    assert messages[-1] == STEERING + "Use a different approach"
    assert ("may or may not have run" in messages[-2]) is not repeated


def test_cancelled_task_stops_without_inference(storage):
    task = storage.add_task()
    storage.tasks[task.id] = task.model_copy(update={"status": "cancelled"})
    storage.tasks[task.id] = storage.tasks[task.id].model_copy(
        update={"status": "running"}
    )
    tasks = ComputerTasks(storage)
    tasks.inference = Commands()

    async def cancel_on_first_read(key, task_id):
        storage.tasks[task_id] = storage.tasks[task_id].model_copy(
            update={"status": "cancelled"}
        )
        return storage.tasks[task_id]

    tasks.task = cancel_on_first_read
    asyncio.run(tasks.serve("test"))
    assert tasks.inference.calls == []
    end = json.loads(contents(storage)[-1].removeprefix(TASK_END))
    assert end["status"] == "cancelled"


def test_inference_failure_fails_the_task(storage):
    task = storage.add_task()
    serve(storage, ServiceError("Hikari is unavailable.", 503))
    done = storage.tasks[task.id]
    assert (done.status, done.summary) == ("failed", "Hikari is unavailable.")


def test_unknown_tool_fails_the_task_without_running_it(storage):
    task = storage.add_task()
    serve(storage, command("send_message", content="hi", end_run=True))
    assert storage.tasks[task.id].status == "failed"
    assert storage.sandbox.calls == []


def test_another_worker_cannot_take_a_held_thread(storage):
    storage.add_task()
    storage.thread.owner_id = uuid4()
    storage.thread.lease_expires_at = datetime.now(UTC)

    async def refuse(service, method, path, json=None, **kwargs):
        if method == "PUT":
            raise ServiceError("Another worker owns this computer thread", 409)
        return await Storage.json_request(storage, service, method, path, json)

    storage.json_request = refuse
    calls = serve(storage)
    assert calls.calls == []


def block(task_id, size, status="completed"):
    start = {"task_id": task_id, "from": "agent", "instructions": "x"}
    end = {"task_id": task_id, "status": status, "summary": f"Summary {task_id}"}
    return [
        RunMessage(role="user", content=TASK_START + json.dumps(start)),
        RunMessage(role="assistant", content="y" * size),
        RunMessage(role="user", content=TASK_END + json.dumps(end)),
    ]


def test_compaction_folds_the_oldest_finished_tasks_into_summaries():
    transcript = [*block("a", 1000), *block("b", 1000), *block("c", 10)]
    compacted = compact(transcript, 1000)
    assert compacted[0].content.startswith(EARLIER)
    earlier = json.loads(compacted[0].content.removeprefix(EARLIER))
    assert [item["task_id"] for item in earlier] == ["a", "b"]
    assert compacted[1:] == block("c", 10)


def test_compaction_keeps_a_thread_that_fits():
    transcript = block("a", 10)
    assert compact(transcript, 10000) is transcript


def test_missing_instructions_fail_the_task_instead_of_retrying(storage, tmp_path):
    storage.settings.computer_instruction_file = str(tmp_path / "missing.md")
    task = storage.add_task()
    calls = serve(storage)
    assert calls.calls == []
    done = storage.tasks[task.id]
    assert done.status == "failed"
    assert "could not be read" in done.summary


def test_cancellation_during_inference_prevents_dispatch(storage):
    task = storage.add_task()
    tasks = ComputerTasks(storage)

    class CancelDuringInference:
        async def call(self, *args):
            storage.tasks[task.id] = storage.tasks[task.id].model_copy(
                update={"status": "cancelled"}
            )
            return command("bash_exec", command="touch file")

    tasks.inference = CancelDuringInference()
    asyncio.run(tasks.serve("test"))
    assert storage.sandbox.calls == []
    assert (
        json.loads(contents(storage)[-1].removeprefix(TASK_END))["status"]
        == "cancelled"
    )
