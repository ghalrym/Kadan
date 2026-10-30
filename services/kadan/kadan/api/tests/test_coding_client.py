"""Coding control with mocked sandbox HTTP and process output, never Codex."""

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
from kadan.api.pydantic_models.coding import CodingStart
from kadan.api.routes.computer import coding
from kadan.api.sandbox_client import SandboxClient
from kadan.runner.tools.coding import codex_cancel, codex_start, codex_status


def job(**changes):
    job_id = str(uuid4())
    return {
        "id": job_id,
        "request_id": job_id,
        "status": "running",
        "cwd": "/home/gem/project",
        "session_id": None,
        "resume_job_id": None,
        "created_at": "2026-10-30T12:00:00+00:00",
        "finished_at": None,
        "message": "Codex is working.",
        "events": [],
        "next_cursor": 0,
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
    assert argv[:2] == ["python3", "/opt/kadan/coding_jobs.py"]
    assert len(argv) == 3
    return json.loads(base64.urlsafe_b64decode(argv[2]))


def test_start_encodes_prompt_without_shell_evaluation_and_preserves_retry_id():
    result = job()
    api, calls = client(result)
    args = CodingStart(
        request_id=result["id"],
        cwd=result["cwd"],
        prompt="`touch /bad` $(whoami) ' \n 🐈",
    )
    saved = asyncio.run(api.start(args))
    assert str(saved.id) == str(args.request_id)
    assert sent(calls) == {"operation": "start", **args.model_dump(mode="json")}
    assert "private-key" not in json.loads(calls[0].content)["command"]
    assert "touch" not in json.loads(calls[0].content)["command"]
    assert calls[0].headers["authorization"] == "Bearer private-key"


def test_unicode_maximum_fits_shell_argument_and_supervisor_bound():
    result = job()
    api, calls = client(result)
    asyncio.run(
        api.start(
            CodingStart(request_id=result["id"], cwd="/home/gem", prompt="🐈" * 16000)
        )
    )
    assert len(json.loads(calls[0].content)["command"].split()[-1]) < 100000


def test_status_and_job_cursor_only_read_saved_state():
    api, calls = client(
        {
            "setup": {
                "state": "auth_required",
                "message": "Sign in inside the computer.",
            },
            "jobs": [],
        }
    )
    assert asyncio.run(api.status()).setup.state == "auth_required"
    assert sent(calls) == {"operation": "status"}
    result = job()
    api, calls = client(result)
    asyncio.run(api.job(UUID(result["id"]), after=17))
    assert sent(calls)["after"] == 17


def test_cancel_returns_stopping_until_cleanup_finishes():
    saved = job(status="stopping")
    api, calls = client(saved)
    result = asyncio.run(api.cancel(UUID(saved["id"])))
    assert result.status == "stopping"
    assert sent(calls)["operation"] == "cancel"


@pytest.mark.parametrize(
    "envelope",
    [
        {},
        {"data": None},
        {"data": {"status": "running", "exit_code": None, "stderr": "secret"}},
        {"data": {"status": "completed", "exit_code": 1, "stderr": "secret"}},
        {"data": {"status": "completed", "exit_code": 0, "stdout": "secret"}},
        {"data": {"status": "completed", "exit_code": 0, "stdout": "[]"}},
    ],
)
def test_ambiguous_control_results_do_not_retry_or_expose_stderr(envelope):
    api, calls = client(None, envelope=envelope)
    with pytest.raises(ServiceError, match="Refresh Coding") as error:
        asyncio.run(
            api.start(CodingStart(request_id=uuid4(), cwd="/home/gem", prompt="task"))
        )
    assert "secret" not in str(error.value)
    assert len(calls) == 2
    assert calls[1].url.path.endswith("/close")


def test_invalid_job_schema_has_safe_error():
    api, _ = client({"id": "secret"})
    with pytest.raises(ServiceError, match="invalid coding status") as error:
        asyncio.run(api.job(uuid4()))
    assert "secret" not in str(error.value)


def test_supervisor_rejection_is_distinct_from_unconfirmed_submission():
    api, calls = client(
        {"error": "The shared computer already has an active coding job."}
    )
    with pytest.raises(ServiceError) as error:
        asyncio.run(api.cancel(uuid4()))
    assert error.value.code == "coding_rejected"
    assert error.value.status == 409
    assert len(calls) == 2


def test_unconfigured_computer_reports_unavailable_without_command():
    api, calls = client(None)
    api.computer.services.settings.sandbox_url = None
    assert asyncio.run(api.status()).setup.state == "unavailable"
    assert calls == []


@pytest.mark.parametrize(
    "change",
    [
        {"cwd": "relative"},
        {"prompt": ""},
        {"timeout_seconds": True},
        {"timeout_seconds": 3601},
        {"request_id": "wrong"},
        {"unexpected": "x"},
    ],
)
def test_start_rejects_invalid_or_unscoped_inputs(change):
    with pytest.raises(ValidationError):
        CodingStart.model_validate(
            {"request_id": str(uuid4()), "cwd": "/home/gem", "prompt": "task", **change}
        )


def test_only_status_is_replayable_and_tools_use_typed_control():
    assert codex_status.metadata["method"] == "GET"
    assert "method" not in codex_start.metadata
    assert codex_cancel.metadata["method"] == "POST"
    result = job()
    api, calls = client(result)
    saved = asyncio.run(
        codex_status.ainvoke(
            {"job_id": result["id"], "after": 3},
            config={"configurable": {"computer": api.computer}},
        )
    )
    assert saved["id"] == result["id"]
    assert sent(calls) == {"operation": "status", "job_id": result["id"], "after": 3}


def test_api_cancel_uses_same_owned_job_protocol():
    result = job(status="stopping")
    api, calls = client(result)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(services=SimpleNamespace(sandbox=api.computer))
        )
    )
    saved = asyncio.run(coding.cancel_coding(request, UUID(result["id"])))
    assert saved.status == "stopping"
    assert sent(calls)["operation"] == "cancel"


@pytest.mark.parametrize("operation", ["start", "job", "cancel"])
def test_wrong_job_result_is_rejected(operation):
    api, _ = client(job(message="Another conversation's output"))
    requested = uuid4()
    with pytest.raises(ServiceError) as error:
        asyncio.run(
            getattr(api, operation)(
                CodingStart(request_id=requested, prompt="task", cwd="/home/gem")
                if operation == "start"
                else requested
            )
        )
    assert error.value.code == "coding_invalid_response"
    assert "Another conversation" not in str(error.value)
