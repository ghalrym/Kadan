"""Notification policy and recovery use fake inference and storage boundaries."""

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from kadan.api.config import ServiceError
from kadan.api.pydantic_models.agent_runs import AgentRun, RunEvent
from kadan.api.pydantic_models.resources import Message
from kadan.api.pydantic_models.tasks import AssistantTask
from kadan.runner.agent_runs import RunConversation
from kadan.runner.progress_updates import decide
from kadan.runner.tools.conversation import (
    SendMessageArguments,
    SendProgressUpdateArguments,
)

from .test_agent_runs import Boundary


def state():
    now = datetime.now(UTC)
    task_id = uuid4()
    run = AgentRun(
        id=uuid4(),
        agent_key="test",
        conversation_id=uuid4(),
        task_id=task_id,
        trigger_event=RunEvent(
            kind="task",
            content=json.dumps(
                {
                    "task_id": str(task_id),
                    "reason": "The saved timer is due.",
                    "retained_events": [],
                }
            ),
        ),
        version=1,
        created_at=now,
        updated_at=now,
    )
    task = AssistantTask(
        id=task_id,
        agent_key="test",
        conversation_id=run.conversation_id,
        origin_run_id=run.id,
        current_run_id=run.id,
        status="active",
        goal="Prepare the report",
        progress="Still waiting for the result",
        next_step="Inspect the result",
        version=1,
        created_at=now,
        updated_at=now,
    )
    return run, task


def answer(**changes):
    return {
        "answers": [
            {
                "key": "progress_update",
                "type": "Choice",
                "value": "suppress",
                "confidence": 0.99,
                "probabilities": {"send": 0.01, "suppress": 0.99},
                **changes,
            }
        ]
    }


def services(response=None):
    return SimpleNamespace(
        json_request=AsyncMock(
            return_value=response if response is not None else answer()
        ),
        resource=AsyncMock(
            return_value=SimpleNamespace(
                messages=[
                    Message(role="assistant", content="Still waiting for the result.")
                ]
            )
        ),
    )


def evaluate(service, run=None, task=None, **changes):
    if run is None:
        run, task = state()
    return asyncio.run(
        decide(
            service,
            run,
            task,
            "Still waiting for the result.",
            **{"requested_by_user": False, "importance": "routine", **changes},
        )
    )


def test_confident_redundant_optional_timer_notice_can_be_suppressed():
    boundary = services()
    assert evaluate(boundary) == "suppress"
    args, kwargs = boundary.json_request.call_args
    assert args == ("Hikari", "POST", "/v1/decisions")
    assert kwargs["timeout"] == 2
    assert kwargs["json"]["questions"][0]["type"] == "Choice"
    assert (
        json.loads(kwargs["json"]["state"])["last_delivered_update"]
        == "Still waiting for the result."
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"value": "send"},
        {"confidence": 0.97},
        {"confidence": None},
        {"probabilities": {"send": 0.03, "suppress": 0.97}},
        {"probabilities": {"send": 0.8, "suppress": 0.99}},
        {"probabilities": {"suppress": 1.0}},
        {"probabilities": {"send": -1.0, "suppress": 2.0}},
        {"probabilities": {"send": float("nan"), "suppress": 0.99}},
        {"key": "other"},
        {"type": "Noul"},
        {"value": "unknown"},
        {"confidence": "0.99"},
        {"unexpected": True},
    ],
)
def test_uncertain_or_malformed_choice_sends(changes):
    assert evaluate(services(answer(**changes))) == "send"


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"answers": []},
        {"answers": [*answer()["answers"], *answer()["answers"]]},
        {"answers": "suppress"},
    ],
)
def test_malformed_response_sends(response):
    assert evaluate(services(response)) == "send"


@pytest.mark.parametrize(
    "failure",
    [
        ServiceError("Decision model unavailable"),
        TimeoutError(),
        ValueError("Invalid JSON"),
    ],
)
def test_unavailable_or_timed_out_classifier_sends(failure):
    boundary = services()
    boundary.json_request.side_effect = failure
    assert evaluate(boundary) == "send"


def test_deadline_expiry_sends_without_a_long_test_wait(monkeypatch):
    class Expired:
        async def __aenter__(self):
            raise TimeoutError

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(
        "kadan.runner.progress_updates.asyncio.timeout", lambda seconds: Expired()
    )
    boundary = services()
    assert evaluate(boundary) == "send"
    boundary.json_request.assert_not_called()


@pytest.mark.parametrize(
    "case",
    [
        "direct_request",
        "requested_update",
        "important_failure",
        "failed_goal",
        "completed_goal",
        "cancelled_goal",
        "paused_goal",
        "blocker",
        "codex_failure",
        "computer_failure",
        "image_completion",
        "resume",
        "retained_failure",
        "malformed_timer",
        "missing_goal",
        "wrong_goal",
    ],
)
def test_important_or_requested_updates_bypass_even_an_erroneous_suppress_answer(case):
    run, task = state()
    options = {}
    if case == "direct_request":
        run.trigger_message_id = uuid4()
    elif case == "requested_update":
        options["requested_by_user"] = True
    elif case == "important_failure":
        options["importance"] = "important"
    elif case.endswith("_goal") and case not in {"missing_goal", "wrong_goal"}:
        task.status = case.removesuffix("_goal")
    elif case == "blocker":
        task.blocker = "Approval required"
    elif case in {"codex_failure", "computer_failure"}:
        run.trigger_event = RunEvent(
            kind=case.removesuffix("_failure"), content='{"status":"failed"}'
        )
    elif case == "image_completion":
        run.trigger_job_id = uuid4()
    elif case == "resume":
        run.trigger_event.content = json.dumps(
            {"task_id": str(task.id), "reason": "Resumed by the user."}
        )
    elif case == "retained_failure":
        event = json.loads(run.trigger_event.content)
        event["retained_events"] = [{"status": "failed"}]
        run.trigger_event.content = json.dumps(event)
    elif case == "malformed_timer":
        run.trigger_event.content = "not JSON"
    elif case == "missing_goal":
        task = None
    elif case == "wrong_goal":
        task.id = uuid4()
    boundary = services()
    assert evaluate(boundary, run, task, **options) == "send"
    boundary.json_request.assert_not_called()
    boundary.resource.assert_not_called()


def test_large_evidence_sends_without_truncating_or_classifying():
    run, task = state()
    task.progress = "x" * 8000
    boundary = services()
    assert evaluate(boundary, run, task) == "send"
    boundary.json_request.assert_not_called()


def test_cancellation_is_not_converted_into_a_visible_notice():
    boundary = services()
    boundary.json_request.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        evaluate(boundary)


@pytest.fixture
def runner(tmp_path):
    boundary = Boundary(tmp_path / "unused")
    run, task = state()
    run = run.model_copy(
        update={
            "id": boundary.current.id,
            "conversation_id": boundary.current.conversation_id,
        }
    )
    task = task.model_copy(
        update={
            "origin_run_id": run.id,
            "current_run_id": run.id,
            "conversation_id": run.conversation_id,
        }
    )
    boundary.current = run
    boundary.start_owned()
    boundary.controller.tasks = AsyncMock(return_value=[task])
    original = boundary.json_request
    boundary.decisions = []

    async def request(service, method, path, **kwargs):
        if service == "Hikari":
            boundary.decisions.append(kwargs["json"])
            return answer()
        return await original(service, method, path, **kwargs)

    boundary.json_request = request
    return boundary


def optional_notice():
    return SendProgressUpdateArguments(
        content="Still waiting.", requested_by_user=False, importance="routine"
    )


def test_suppression_is_checkpointed_and_replayed_without_another_classifier_call(
    runner,
):
    async def scenario():
        first = await RunConversation(
            runner.controller, runner.current
        ).send_progress_update(optional_notice())
        saved = runner.current
        assert list(saved.progress_update_decisions.values()) == ["suppress"]
        replay = await RunConversation(
            runner.controller, saved, replayed=True
        ).send_progress_update(optional_notice())
        assert (
            first
            == replay
            == {"tool": "send_progress_update", "delivered": False, "suppressed": True}
        )

    asyncio.run(scenario())
    assert len(runner.decisions) == 1
    assert runner.deliveries == []


def test_delivery_replay_uses_saved_send_and_one_message_identity(runner):
    runner.crash_after_delivery = True
    args = SendProgressUpdateArguments(content="An important result.")

    async def scenario():
        with pytest.raises(asyncio.CancelledError):
            await RunConversation(
                runner.controller, runner.current
            ).send_progress_update(args)
        assert list(runner.current.progress_update_decisions.values()) == ["send"]
        await RunConversation(
            runner.controller, runner.current, replayed=True
        ).send_progress_update(args)

    asyncio.run(scenario())
    assert len(runner.deliveries) == 1
    assert len(set(runner.delivery_attempts)) == 1
    assert runner.decisions == []


def test_cancel_during_classification_prevents_receipt_and_delivery(runner):
    original = runner.json_request

    async def request(service, *args, **kwargs):
        if service == "Hikari":
            runner.cancel()
            return answer()
        return await original(service, *args, **kwargs)

    runner.json_request = request
    with pytest.raises(ServiceError):
        asyncio.run(
            RunConversation(runner.controller, runner.current).send_progress_update(
                optional_notice()
            )
        )
    assert runner.current.progress_update_decisions == {}
    assert runner.deliveries == []


def test_ordinary_message_never_uses_classifier(runner):
    result = asyncio.run(
        RunConversation(runner.controller, runner.current).send_message(
            SendMessageArguments(content="A requested reply.", end_run=True)
        )
    )
    assert result["delivered"] is True
    assert runner.decisions == []
    assert runner.current.progress_update_decisions == {}


def test_progress_arguments_default_to_unsuppressible():
    args = SendProgressUpdateArguments(content="Please update me.")
    assert args.requested_by_user is True
    assert args.importance == "important"
    with pytest.raises(ValidationError):
        SendProgressUpdateArguments(content="Update", requested_by_user="false")
