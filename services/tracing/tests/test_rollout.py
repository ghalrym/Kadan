import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
from kadan_tracing.rollout import Admission, RolloutAdmission, smoke


def gate_at(tmp_path):
    fence = tmp_path / "fence.json"
    fence.write_text(json.dumps({"closed": []}))
    return Admission("test-api", fence)


def test_fence_persists_for_new_process_and_fails_closed(tmp_path):
    gate = gate_at(tmp_path)
    assert gate.enter()
    gate.fence.write_text('{"closed":["test-api"]}')
    assert not gate.enter()
    assert gate.status()["active"] == 1
    assert Admission("test-api", gate.fence).fenced
    gate.leave()
    gate.fence.write_text("broken")
    assert gate.fenced
    gate.fence.unlink()
    assert gate.fenced


def test_admitted_http_response_finishes_while_new_requests_are_rejected(tmp_path):
    async def scenario():
        gate = gate_at(tmp_path)
        entered, finish = asyncio.Event(), asyncio.Event()
        messages = []

        async def app(scope, receive, send):
            entered.set()
            await finish.wait()
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"done"})

        async def send(message):
            messages.append(message)

        wrapped = RolloutAdmission(app, gate)
        first = asyncio.create_task(
            wrapped({"type": "http", "path": "/work"}, None, send)
        )
        await entered.wait()
        gate.fence.write_text('{"closed":["test-api"]}')
        await wrapped({"type": "http", "path": "/work"}, None, send)
        assert messages[0]["status"] == 503
        assert gate.status()["active"] == 1
        finish.set()
        await first
        assert messages[2]["status"] == 200
        assert gate.status()["active"] == 0

    asyncio.run(scenario())


def test_http_failure_releases_admission(tmp_path):
    async def scenario():
        gate = gate_at(tmp_path)

        async def app(*args):
            raise ValueError("failed")

        with pytest.raises(ValueError):
            await RolloutAdmission(app, gate)(
                {"type": "http", "path": "/work"}, None, None
            )
        assert gate.active == 0

    asyncio.run(scenario())


def test_native_work_and_readiness_are_part_of_status(tmp_path):
    gate = gate_at(tmp_path)
    gate.extra_active = lambda: 2
    gate.check_ready = lambda: False
    assert gate.status() == {"fenced": False, "active": 2, "ready": False}


@pytest.mark.parametrize("path", ["/health", "/api/health"])
def test_health_remains_available_while_fenced(tmp_path, path):
    async def scenario():
        gate = gate_at(tmp_path)
        gate.fence.write_text('{"closed":["test-api"]}')
        called = []

        async def app(*args):
            called.append(True)

        await RolloutAdmission(app, gate)({"type": "http", "path": path}, None, None)
        assert called == [True]
        assert gate.active == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "service,expected",
    [("kadan-api", 3), ("kadan-runner", 2), ("tsuchi-api", 1), ("hikari-api", 1)],
)
def test_smoke_checks_fixed_health_endpoints_without_proxies(
    monkeypatch, service, expected
):
    monkeypatch.setenv("KADAN_ROLLOUT_SERVICE", service)
    response = MagicMock()
    response.__enter__.return_value = response
    response.status = 200
    response.read.return_value = b'{"status":"ok"}'
    with patch("kadan_tracing.rollout.build_opener") as build:
        build.return_value.open.return_value = response
        assert smoke() == {"ready": True}
        assert build.call_args.args[0].proxies == {}
        assert build.return_value.open.call_count == expected
        for call in build.return_value.open.call_args_list:
            assert call.args[0].endswith("/health")
            assert call.kwargs == {"timeout": 3}


def test_smoke_rejects_unhealthy_response(monkeypatch):
    monkeypatch.setenv("KADAN_ROLLOUT_SERVICE", "tsuchi-api")
    response = MagicMock()
    response.__enter__.return_value = response
    response.status = 200
    response.read.return_value = b'{"status":"failed"}'
    with patch("kadan_tracing.rollout.build_opener") as build:
        build.return_value.open.return_value = response
        with pytest.raises(ValueError, match="health check failed"):
            smoke()


def test_smoke_refuses_unknown_service(monkeypatch):
    monkeypatch.setenv("KADAN_ROLLOUT_SERVICE", "unknown")
    with pytest.raises(ValueError, match="not enrolled"):
        smoke()
