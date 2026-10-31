import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import Request
from pydantic import ValidationError

from kadan.api.routes.agents import computer


def task(instructions="Open the docs"):
    now = datetime.now(UTC)
    return {
        "id": str(uuid4()),
        "agent_key": "andrew",
        "instructions": instructions,
        "status": "running",
        "version": 1,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }


def request(responses):
    calls = []

    async def json_request(service, method, path, **kwargs):
        calls.append((service, method, path, kwargs.get("json")))
        return responses[(method, path)]

    req = Request(
        {
            "type": "http",
            "app": SimpleNamespace(
                state=SimpleNamespace(
                    services=SimpleNamespace(json_request=json_request)
                )
            ),
        }
    )
    return req, calls


def test_work_reports_the_current_task_without_the_transcript():
    running, queued = task(), task("Then close them")
    thread = {
        "agent_key": "andrew",
        "transcript": [{"role": "user", "content": "private"}],
        "steering": [
            {
                "id": str(uuid4()),
                "task_id": running["id"],
                "content": "faster",
                "created_at": running["created_at"],
            }
        ],
        "version": 3,
        "updated_at": running["updated_at"],
    }
    req, _ = request(
        {
            ("GET", "/api/agents/andrew/computer-tasks"): [running, queued],
            ("GET", "/api/agents/andrew/computer-thread"): thread,
        }
    )
    work = asyncio.run(computer.computer_work("andrew", req))
    assert str(work.task.id) == running["id"]
    assert (work.queued, work.pending_steering) == (1, 1)
    assert "transcript" not in work.model_dump()


def test_idle_computer_reports_its_last_result():
    finished = {**task(), "status": "failed", "summary": "No network."}
    req, _ = request(
        {
            ("GET", "/api/agents/andrew/computer-tasks"): [],
            ("GET", "/api/agents/andrew/computer-tasks/latest"): finished,
            ("GET", "/api/agents/andrew/computer-thread"): {
                "agent_key": "andrew",
                "version": 1,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        }
    )
    work = asyncio.run(computer.computer_work("andrew", req))
    assert (work.task, work.queued, work.pending_steering) == (None, 0, 0)
    assert work.last.summary == "No network."


def test_request_is_forwarded_to_storage():
    value = computer.ComputerRequestInput(id=uuid4(), content="Open a terminal")
    req, calls = request(
        {
            ("POST", "/api/agents/andrew/computer-requests"): {
                "kind": "steering",
                "task": task(),
            }
        }
    )
    result = asyncio.run(computer.computer_request("andrew", value, req))
    assert result.kind == "steering"
    assert calls[0][3] == {"id": str(value.id), "content": "Open a terminal"}


def test_blank_request_is_rejected():
    with pytest.raises(ValidationError):
        computer.ComputerRequestInput(id=uuid4(), content="   ")
