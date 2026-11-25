"""The computer agent's worker, with in-memory storage, computer and inference."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from kadan.api.config import ServiceError
from kadan.api.pydantic_models.agent_runs import (
    RunCall,
    RunMessage,
    TaskEnd,
    TaskStart,
    ToolCommand,
    UnrecoveredRecord,
)
from kadan.api.pydantic_models.computer import (
    ComputerSteering,
    ComputerTask,
    ComputerTaskOrigin,
    ComputerThread,
)
from kadan.runner.computer_tasks import (
    NEW_STEERING,
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
        self.priorities = []

    async def call(
        self, messages, model_id, conversation_key, tools, *, priority="interactive", **kwargs
    ):
        self.calls.append(messages)
        self.priorities.append(priority)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result if isinstance(result, RunCall) else RunCall(command=result)


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


def test_task_runs_computer_tools_then_reports_its_summary(storage):
    task = storage.add_task()
    selected = RunCall(
        call_id="provider_computer_7", command=command("bash_exec", command="echo hi")
    )
    calls = serve(storage, selected, finish("Printed hi."))
    assert calls.calls[1][-1].tool_call_id == selected.call_id
    assert calls.calls[1][-2].tool_calls[0]["id"] == selected.call_id

    method, path, request = storage.sandbox.calls[0]
    assert (method, path) == ("POST", "/v1/bash/exec")
    assert request["json_body"]["command"] == "echo hi"
    done = storage.tasks[task.id]
    assert (done.status, done.summary) == ("completed", "Printed hi.")
    assert storage.events == [done]
    assert storage.thread.transcript[0].role == "task_start"
    assert storage.thread.transcript[-1].status == ("completed")
    assert storage.thread.owner_id is None


def test_unsuccessful_finish_fails_the_task(storage):
    task = storage.add_task()
    serve(storage, finish("No such site.", succeeded=False))
    assert storage.tasks[task.id].status == "failed"


def test_user_steering_is_read_before_the_next_inference(storage):
    storage.add_task(origin=False)
    storage.steer("Use the terminal instead")
    calls = serve(storage, finish())
    assert calls.calls[0][-1].content == "Use the terminal instead"
    assert storage.steering == []


def test_steering_that_arrives_before_finishing_keeps_the_task_going(storage):
    task = storage.add_task()
    storage.arrive_before_finish = ["Also open the changelog"]
    calls = serve(storage, finish("Docs open."), finish("Docs and changelog open."))
    assert len(calls.calls) == 2
    second = [message.content for message in calls.calls[1]]
    assert any("new steering" in item for item in second)
    assert second[-1] == "Also open the changelog"
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
    ("tool", "repeated"), [("bash_exec", False), ("browser_get_text", True)]
)
def test_restart_never_repeats_an_unconfirmed_side_effect(storage, tool, repeated):
    task = storage.add_task()
    task = storage.tasks[task.id] = task.model_copy(update={"status": "running"})
    start = TaskStart(task_id=task.id, origin="agent", instructions="Do it")
    pending = command(tool, **({"command": "ls"} if tool == "bash_exec" else {}))
    storage.thread.transcript = [
        start,
        RunCall(command=pending),
    ]
    storage.steer("Use a different approach")
    calls = serve(storage, finish())
    assert bool(storage.sandbox.calls) is repeated
    messages = [item.content for item in calls.calls[0]]
    assert messages[-1] == "Use a different approach"
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
    assert storage.thread.transcript[-1].status == "cancelled"


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
    ident = UUID(int=ord(task_id))
    start = TaskStart(task_id=ident, origin="agent", instructions="x")
    end = TaskEnd(task_id=ident, status=status, summary=f"Summary {task_id}")
    return [
        start,
        RunMessage(role="assistant", content="y" * size),
        end,
    ]


def test_compaction_folds_the_oldest_finished_tasks_into_summaries():
    transcript = [*block("a", 1000), *block("b", 1000), *block("c", 10)]
    compacted = compact(transcript, 1000)
    assert compacted[0].role == "task_history"
    earlier = compacted[0].tasks
    assert [item.task_id.int for item in earlier] == [ord("a"), ord("b")]
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
        async def call(self, *args, **kwargs):
            storage.tasks[task.id] = storage.tasks[task.id].model_copy(
                update={"status": "cancelled"}
            )
            return RunCall(command=command("bash_exec", command="touch file"))

    tasks.inference = CancelDuringInference()
    asyncio.run(tasks.serve("test"))
    assert storage.sandbox.calls == []
    assert storage.thread.transcript[-1].status == "cancelled"


def test_ambiguous_saved_task_fails_before_replay_and_next_task_can_continue(storage):
    first = storage.add_task("first")
    storage.tasks[first.id] = first.model_copy(update={"status": "running"})
    storage.thread.transcript = [
        UnrecoveredRecord(original={"role": "assistant", "content": "ambiguous action"})
    ]
    second = storage.add_task("second")
    calls = serve(storage, finish("Second completed."))
    assert storage.tasks[first.id].status == "failed"
    assert "could not safely resume" in storage.tasks[first.id].summary
    assert storage.tasks[second.id].status == "completed"
    assert len(calls.calls) == 1


def test_finished_tasks_are_folded_and_computer_uses_background_priority(storage):
    storage.thread.transcript = [
        *block("a", 5000, status="failed"),
        *block("b", 5000, status="failed"),
        *block("c", 5000, status="failed"),
    ]
    storage.add_task("Fresh task")
    calls = serve(storage, finish("Fresh done."))
    assert calls.priorities == ["background"]
    prompt_chars = sum(len(str(msg.content)) for msg in calls.calls[0])
    assert prompt_chars < 2000
    assert storage.thread.transcript[0].role == "task_history"
    assert len(storage.thread.transcript[0].tasks) == 3
