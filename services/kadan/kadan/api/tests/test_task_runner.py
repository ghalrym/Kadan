"""Goal orchestration with recording storage and inference boundaries."""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from pydantic import ValidationError

from kadan.api.config import ServiceError
from kadan.api.pydantic_models.tasks import AssistantTask
from kadan.runner.agent_runs import RunConversation
from kadan.runner.tools.goals import CheckpointTaskArguments

from .test_agent_runs import Commands, boundary, command  # noqa: F401


@pytest.fixture
def goals(boundary):  # noqa: F811 - pytest resolves the imported boundary fixture.
    boundary.saved_tasks = {}
    boundary.checkpoints = {}
    boundary.task_requests = []
    original = boundary.json_request

    async def request(service, method, path, **kwargs):
        if path.endswith("/tasks"):
            return [
                task.model_dump(mode="json") for task in boundary.saved_tasks.values()
            ]
        if path.endswith("/task") and method == "POST":
            body = kwargs["json"]
            boundary.check_owned(body)
            boundary.task_requests.append(body)
            action_id = UUID(body["action_id"])
            if action_id not in boundary.checkpoints:
                now = datetime.now(UTC)
                task_id = UUID(body["task_id"]) if body["task_id"] else action_id
                old = boundary.saved_tasks.get(task_id)
                if old and old.version != body["expected_version"]:
                    raise ServiceError("Goal changed", 409)
                task = AssistantTask(
                    id=task_id,
                    agent_key="test",
                    conversation_id=boundary.conversation.id,
                    origin_run_id=boundary.current.id,
                    current_run_id=boundary.current.id,
                    version=old.version + 1 if old else 1,
                    created_at=now,
                    updated_at=now,
                    **{
                        name: body[name]
                        for name in [
                            "goal",
                            "progress",
                            "next_step",
                            "blocker",
                            "status",
                            "wait_kind",
                            "wake_at",
                        ]
                    },
                )
                boundary.saved_tasks[task_id] = task
                boundary.checkpoints[action_id] = task
                boundary.current = boundary.current.model_copy(
                    update={"task_id": task_id, "version": boundary.current.version + 1}
                )
            return boundary.checkpoints[action_id].model_dump(mode="json")
        return await original(service, method, path, **kwargs)

    boundary.json_request = request
    return boundary


def checkpoint(**changes):
    return command(
        "checkpoint_task",
        goal="Prepare a reviewable change",
        next_step="Inspect the result",
        status="active",
        **changes,
    )


def test_goal_checkpoint_then_delegation_records_tool_wait(goals):
    goals.sandbox.configured = True
    goals.controller.inference = Commands(
        checkpoint(),
        command(
            "use_computer",
            instructions="Inspect the requested file",
            message="Checking the file.",
        ),
    )
    asyncio.run(goals.controller.process(goals.current))
    assert goals.current.status == "completed"
    assert len(goals.computer_tasks) == 1
    task = next(iter(goals.saved_tasks.values()))
    assert (task.status, task.wait_kind) == ("waiting", "tool")
    assert task.goal == "Prepare a reviewable change"
    assert len(goals.controller.inference.calls) == 2


def test_waiting_for_user_ends_response_without_another_model_call(goals):
    goals.controller.inference = Commands(
        command(
            "checkpoint_task",
            goal="Choose the target",
            next_step="Use the chosen target",
            status="waiting",
            wait_kind="user",
            blocker="Target needed",
            message="Which target should I use?",
        ),
    )
    asyncio.run(goals.controller.process(goals.current))
    task = next(iter(goals.saved_tasks.values()))
    assert (task.status, task.wait_kind) == ("waiting", "user")
    assert goals.task_requests[0]["message"] == "Which target should I use?"
    assert goals.current.status == "completed"
    assert len(goals.controller.inference.calls) == 1


def test_create_checkpoint_replay_retains_original_identity(goals):
    async def scenario():
        run = goals.start_owned()
        args = CheckpointTaskArguments(
            goal="Saved goal", next_step="Continue", status="active"
        )
        first = await RunConversation(goals.controller, run).checkpoint_task(args)
        replay = await RunConversation(
            goals.controller, goals.current, replayed=True
        ).checkpoint_task(args)
        assert first == replay

    asyncio.run(scenario())
    assert len(goals.saved_tasks) == 1
    assert [request["task_id"] for request in goals.task_requests] == [None, None]
    assert len({request["action_id"] for request in goals.task_requests}) == 1


def test_timer_checkpoint_is_saved_without_waiting_in_process(goals):
    due = datetime.now(UTC) + timedelta(minutes=5)
    goals.controller.inference = Commands(
        command(
            "checkpoint_task",
            goal="Check later",
            next_step="Inspect the new result",
            status="waiting",
            wait_kind="timer",
            wake_at=due,
        )
    )
    asyncio.run(goals.controller.process(goals.current))
    task = next(iter(goals.saved_tasks.values()))
    assert (task.wait_kind, task.wake_at) == ("timer", due)
    assert goals.current.status == "completed"


def test_saved_goal_is_in_fresh_context_outside_recent_chat(goals):
    async def scenario():
        run = goals.start_owned()
        await RunConversation(goals.controller, run).checkpoint_task(
            CheckpointTaskArguments(
                goal="Remember durable goal",
                progress="File verified",
                next_step="Review result",
                status="active",
            )
        )
        messages = await goals.controller.context(goals.current)
        assert "Remember durable goal" in messages[-1].content
        assert "File verified" in messages[-1].content

    asyncio.run(scenario())


def test_cancelled_goal_cannot_delegate_after_checkpoint(goals):
    async def scenario():
        run = goals.start_owned()
        await RunConversation(goals.controller, run).checkpoint_task(
            CheckpointTaskArguments(
                goal="Saved goal", next_step="Continue", status="active"
            )
        )
        goals.cancel()
        with pytest.raises(ServiceError):
            await RunConversation(goals.controller, goals.current).wait_for_tool()
        assert not goals.computer_tasks

    asyncio.run(scenario())


def test_checkpoint_does_not_race_heartbeat_version_update(goals):
    async def scenario():
        run = goals.start_owned()
        read = asyncio.Event()
        release = asyncio.Event()
        original = goals.json_request
        blocked = False

        async def request(service, method, path, **kwargs):
            nonlocal blocked
            if (
                method == "GET"
                and path == goals.controller.path("test", run.id)
                and not blocked
            ):
                blocked = True
                value = await original(service, method, path, **kwargs)
                read.set()
                await release.wait()
                return value
            return await original(service, method, path, **kwargs)

        goals.json_request = request
        heartbeat = asyncio.create_task(
            goals.controller.update(
                run, lease_expires_at=datetime.now(UTC) + timedelta(seconds=60)
            )
        )
        await read.wait()
        checkpoint = asyncio.create_task(
            RunConversation(goals.controller, run).checkpoint_task(
                CheckpointTaskArguments(
                    goal="Saved goal", next_step="Continue", status="active"
                )
            )
        )
        await asyncio.sleep(0)
        assert not goals.task_requests
        release.set()
        await asyncio.gather(heartbeat, checkpoint)
        assert goals.current.status == "running"
        assert len(goals.saved_tasks) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "waiting", "wait_kind": "user", "blocker": "Choose a target"},
        {"status": "waiting", "wait_kind": "timer", "wake_at": datetime(2026, 11, 2)},  # noqa: DTZ001 - validates rejection of naive times.
        {"status": "failed"},
        {"status": "active", "next_step": ""},
    ],
)
def test_invalid_goal_states_are_rejected_before_execution(changes):
    with pytest.raises(ValidationError):
        CheckpointTaskArguments.model_validate(
            {"goal": "A goal", "next_step": "Continue", **changes}
        )
