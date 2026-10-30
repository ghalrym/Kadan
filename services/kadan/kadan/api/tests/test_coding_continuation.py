"""Explicit continuation submission with mocked storage and computer boundaries."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from kadan.api.config import ServiceError
from kadan.api.pydantic_models.agent_runs import AgentRun
from kadan.api.pydantic_models.coding import CodingJob
from kadan.api.routes.agents.conversations.runs import RunInput
from kadan.api.run_client import RunClient


@pytest.fixture
def submission(monkeypatch):
    now = datetime.now(UTC)
    job = CodingJob(
        id=uuid4(),
        request_id=uuid4(),
        status="completed",
        cwd="/home/gem/project",
        session_id=str(uuid4()),
        created_at=now.isoformat(),
        message="Earlier turn completed",
    )
    job.request_id = job.id
    value = RunInput(
        content="Rerun the tests",
        expected_version=2,
        persona_id=None,
        resume_coding_job_id=job.id,
    )
    conversation_id = uuid4()
    computer = SimpleNamespace(job=AsyncMock(return_value=job))
    monkeypatch.setattr("kadan.api.run_client.CodingClient", lambda sandbox: computer)

    async def enqueue(service, method, path, *, json):
        assert (service, method, path) == (
            "Tsuchi",
            "POST",
            f"/api/agents/test/resources/{conversation_id}/runs",
        )
        return AgentRun(
            id=value.message_id,
            agent_key="test",
            conversation_id=conversation_id,
            trigger_message_id=value.message_id,
            coding_task=json.get("coding_task"),
            version=1,
            created_at=now,
            updated_at=now,
        ).model_dump(mode="json")

    services = SimpleNamespace(
        sandbox=object(), json_request=AsyncMock(side_effect=enqueue)
    )
    return SimpleNamespace(
        job=job,
        value=value,
        conversation_id=conversation_id,
        computer=computer,
        services=services,
        client=RunClient(services),
    )


def test_submission_resolves_directory_from_saved_job_and_persists_explicit_intent(
    submission,
):
    s = submission
    run = asyncio.run(s.client.enqueue("test", s.conversation_id, s.value))
    s.computer.job.assert_awaited_once_with(s.job.id)
    task = run.coding_task
    assert task.cwd == s.job.cwd and task.resume_job_id == s.job.id
    assert task.prompt == s.value.content
    body = s.services.json_request.call_args.kwargs["json"]
    assert body["message"]["id"] == str(s.value.message_id)
    assert body["message"]["content"].startswith(
        "Continue with Codex in `/home/gem/project`:"
    )
    assert body["coding_task"] == task.model_dump(mode="json")


@pytest.mark.parametrize(
    "status", ["starting", "running", "stopping", "cleanup_required"]
)
def test_unfinished_or_unconfirmed_cleanup_never_queues_a_continuation(
    submission, status
):
    s = submission
    s.job.status = status
    with pytest.raises(ServiceError, match="cleanup"):
        asyncio.run(s.client.enqueue("test", s.conversation_id, s.value))
    s.services.json_request.assert_not_awaited()


def test_missing_session_never_queues_or_falls_back_to_a_new_session(submission):
    s = submission
    s.job.session_id = None
    with pytest.raises(ServiceError, match="no saved session"):
        asyncio.run(s.client.enqueue("test", s.conversation_id, s.value))
    s.services.json_request.assert_not_awaited()


def test_storage_must_confirm_the_same_conversation_and_intent(submission):
    s = submission
    original = s.services.json_request.side_effect

    async def wrong(*args, **kwargs):
        run = await original(*args, **kwargs)
        run["coding_task"] = None
        return run

    s.services.json_request.side_effect = wrong
    with pytest.raises(ServiceError, match="did not confirm"):
        asyncio.run(s.client.enqueue("test", s.conversation_id, s.value))


def test_plain_chat_submission_keeps_existing_behavior_without_computer_reads(
    submission,
):
    s = submission
    s.value.resume_coding_job_id = None
    run = asyncio.run(s.client.enqueue("test", s.conversation_id, s.value))
    assert run.coding_task is None
    s.computer.job.assert_not_awaited()
    body = s.services.json_request.call_args.kwargs["json"]
    assert "coding_task" not in body and body["message"]["content"] == s.value.content


def test_client_cannot_supply_its_own_coding_directory_or_action(submission):
    value = submission.value.model_dump()
    with pytest.raises(ValidationError):
        RunInput.model_validate({**value, "coding_task": {"cwd": "/elsewhere"}})


def test_older_storage_rejection_has_actionable_update_guidance(submission):
    s = submission
    s.services.json_request.side_effect = ServiceError("Tsuchi returned HTTP 422", 422)
    with pytest.raises(ServiceError, match="updated together") as caught:
        asyncio.run(s.client.enqueue("test", s.conversation_id, s.value))
    assert caught.value.status == 503


def test_followup_limit_is_validated_at_request_boundary_without_reducing_chat_limit(
    submission,
):
    value = submission.value.model_dump()
    assert (
        len(RunInput.model_validate({**value, "content": "x" * 16000}).content) == 16000
    )
    with pytest.raises(ValidationError, match="16000"):
        RunInput.model_validate({**value, "content": "x" * 16001})
    chat = {**value, "content": "x" * 32000, "resume_coding_job_id": None}
    assert len(RunInput.model_validate(chat).content) == 32000
