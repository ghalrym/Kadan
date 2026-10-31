"""Codex control with mocked sandbox HTTP, never Codex."""

import asyncio
import base64
import json
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from kadan.api.coding_client import CodingClient
from kadan.api.config import ServiceError
from kadan.api.pydantic_models.coding import CodexOrigin, CodexTask
from kadan.api.routes.computer import coding
from kadan.api.sandbox_client import SandboxClient
from kadan.runner.tools.coding import codex_start, codex_stop, codex_threads


def summary(**changes):
    return {
        "id": str(uuid4()),
        "title": "Fix the parser",
        "cwd": "/home/gem/project",
        "created_at": "2026-10-30T12:00:00+00:00",
        "updated_at": "2026-10-30T12:05:00+00:00",
        "running": True,
        **changes,
    }


def client(result, *, envelope=None):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(
            200,
            json=envelope
            if envelope is not None
            else {
                "data": {
                    "status": "completed",
                    "exit_code": 0,
                    "stdout": json.dumps(result),
                },
            },
        )

    computer = SandboxClient(
        SimpleNamespace(
            client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
            settings=SimpleNamespace(
                sandbox_url="http://computer", sandbox_api_key=SecretStr("private-key")
            ),
        )
    )
    return CodingClient(computer), calls


def sent(calls):
    request = json.loads(calls[0].content)
    argv = request["command"].split()
    assert argv[:2] == ["python3", "/opt/kadan/codex_threads.py"]
    assert len(argv) == 3
    return json.loads(base64.urlsafe_b64decode(argv[2]))


def route_request(computer):
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(services=SimpleNamespace(sandbox=computer))
        )
    )


def test_start_encodes_prompt_without_shell_evaluation():
    thread_id = str(uuid4())
    api, calls = client(
        {"thread_id": thread_id, "cwd": "/home/gem/project", "running": True}
    )
    prompt = "Fix it'; rm -rf / #\n$(whoami)"
    request_id = uuid4()
    origin = CodexOrigin(agent_key="guide", conversation_id=uuid4(), run_id=uuid4())
    started = asyncio.run(
        api.start(CodexTask(prompt=prompt, cwd="/home/gem/project"), request_id, origin)
    )
    assert str(started.thread_id) == thread_id
    assert sent(calls) == {
        "operation": "start",
        "prompt": prompt,
        "cwd": "/home/gem/project",
        "thread_id": None,
        "request_id": str(request_id),
        "origin": origin.model_dump(mode="json"),
    }
    assert calls[1].url.path.endswith("/close")


@pytest.mark.parametrize(
    "envelope",
    [
        {"data": {"status": "completed", "exit_code": 1, "stdout": "secret"}},
        {"data": {"status": "completed", "exit_code": 0, "stdout": "secret"}},
        {"data": {"status": "completed", "exit_code": 0, "stdout": "[]"}},
    ],
)
def test_unreadable_results_are_unavailable_without_output(envelope):
    api, calls = client(None, envelope=envelope)
    with pytest.raises(ServiceError) as error:
        asyncio.run(api.status())
    assert error.value.code == "codex_unavailable"
    assert "secret" not in str(error.value)
    assert len(calls) == 2


def test_script_rejection_is_reported_as_conflict():
    api, _ = client({"error": "That thread is already running."})
    with pytest.raises(ServiceError) as error:
        asyncio.run(api.stop(uuid4()))
    assert error.value.code == "codex_rejected"
    assert error.value.status == 409


def test_invalid_schema_has_safe_error():
    api, _ = client({"id": "secret"})
    with pytest.raises(ServiceError, match="invalid Codex response") as error:
        asyncio.run(api.thread(uuid4()))
    assert "secret" not in str(error.value)


def test_unconfigured_computer_reports_unavailable_without_command():
    api, calls = client(None)
    api.computer.services.settings.sandbox_url = None
    assert asyncio.run(api.login_status()).state == "unavailable"
    assert calls == []


@pytest.mark.parametrize(
    "change", [{"cwd": "relative"}, {"prompt": ""}, {"thread_id": "x"}, {"other": 1}]
)
def test_task_rejects_invalid_inputs(change):
    with pytest.raises(ValidationError):
        CodexTask.model_validate({"cwd": "/home/gem", "prompt": "task", **change})


def test_login_route_returns_device_code():
    api, calls = client(
        {
            "state": "signing_in",
            "message": "Open the sign-in page and enter the code.",
            "login": {"url": "https://auth.example/device", "code": "ABCD-1234"},
        }
    )
    setup = asyncio.run(coding.codex_login(route_request(api.computer)))
    assert setup.login.code == "ABCD-1234"
    assert sent(calls) == {"operation": "login"}


def test_stop_route_sends_thread_id():
    thread_id = uuid4()
    api, calls = client({"thread_id": str(thread_id), "running": False})
    stopped = asyncio.run(
        coding.stop_codex_thread(route_request(api.computer), thread_id)
    )
    assert stopped.running is False
    assert sent(calls) == {"operation": "stop", "thread_id": str(thread_id)}


def test_only_thread_reads_are_replayable():
    assert codex_threads.metadata["method"] == "GET"
    assert codex_stop.metadata["method"] == "POST"
    # Start replays safely because the computer keys it by the action ID.
    assert "method" not in codex_start.metadata


def test_finished_runs_and_acknowledgement_use_the_script():
    finished = {
        "request_id": str(uuid4()),
        "thread_id": str(uuid4()),
        "cwd": "/home/gem/project",
        "origin": None,
        "status": "completed",
        "message": "Done.",
    }
    api, calls = client({"runs": [finished]})
    runs = asyncio.run(api.finished())
    assert runs[0].message == "Done."
    assert sent(calls) == {"operation": "finished"}
    api, calls = client({"request_id": finished["request_id"]})
    asyncio.run(api.acknowledge(UUID(finished["request_id"])))
    assert sent(calls) == {
        "operation": "acknowledge",
        "request_id": finished["request_id"],
    }


def test_threads_tool_reads_latest_items():
    thread = {
        **summary(),
        "items": [{"kind": "user", "text": f"step {index}"} for index in range(5)],
        "items_truncated": False,
    }
    api, calls = client(thread)
    result = asyncio.run(
        codex_threads.ainvoke(
            {"thread_id": thread["id"], "last": 2},
            config={"configurable": {"computer": api.computer}},
        )
    )
    assert [item["text"] for item in result["items"]] == ["step 3", "step 4"]
    assert result["running"] is True
    assert sent(calls) == {"operation": "thread", "thread_id": thread["id"]}
    assert UUID(result["id"])
