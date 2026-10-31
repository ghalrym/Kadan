import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi import WebSocketDisconnect

from kadan.api.config import ServiceError
from kadan.api.routes import calls

PERSONA_ID = UUID("10000000-0000-4000-8000-000000000002")


def socket(url="http://hikari-api:8000", persona_id=None):
    value = AsyncMock()
    value.receive.return_value = {
        "type": "websocket.receive",
        "text": json.dumps({"type": "start", "persona_id": persona_id}),
    }
    value.app = SimpleNamespace(
        state=SimpleNamespace(
            services=SimpleNamespace(
                settings=SimpleNamespace(hikari_url=url),
                call_instruction=AsyncMock(return_value="Saved persona instructions."),
            )
        )
    )
    return value


def test_call_resolves_persona_before_connecting_and_relays_after_sending_start(
    monkeypatch,
):
    browser = socket("https://hikari.local/prefix", str(PERSONA_ID))
    upstream = AsyncMock()

    async def connect_call(*args, **kwargs):
        browser.app.state.services.call_instruction.assert_awaited_once_with(
            "test", PERSONA_ID
        )
        return upstream

    connect = AsyncMock(side_effect=connect_call)
    relay = AsyncMock()
    monkeypatch.setattr(calls, "connect", connect)
    monkeypatch.setattr(calls, "relay", relay)
    asyncio.run(calls.voice_call("test", browser))
    connect.assert_awaited_once_with(
        "wss://hikari.local/prefix/v1/audio/calls",
        open_timeout=5,
        max_size=6 * 1024**2,
        proxy=None,
    )
    assert json.loads(upstream.send.call_args.args[0]) == {
        "type": "start",
        "instruction": "Saved persona instructions.",
    }
    relay.assert_awaited_once_with(browser, upstream)
    # The generic relay owns closure after handoff.
    upstream.close.assert_not_awaited()


@pytest.mark.parametrize("send_fails", [False, True])
def test_unavailable_hikari_closes_call_with_actionable_error(monkeypatch, send_fails):
    browser = socket()
    upstream = AsyncMock()
    upstream.send.side_effect = OSError("closed")
    connect = (
        AsyncMock(return_value=upstream)
        if send_fails
        else AsyncMock(side_effect=OSError("refused"))
    )
    monkeypatch.setattr(calls, "connect", connect)
    asyncio.run(calls.voice_call("test", browser))
    assert "Hikari is unavailable" in browser.send_json.call_args.args[0]["message"]
    browser.close.assert_awaited_once_with(1013)
    if send_fails:
        upstream.close.assert_awaited_once()


@pytest.mark.parametrize(
    "message",
    [
        {"type": "websocket.receive", "text": "not json"},
        {"type": "websocket.receive", "bytes": b"audio"},
        {"type": "websocket.receive", "text": json.dumps({"type": "end"})},
        {
            "type": "websocket.receive",
            "text": json.dumps({"type": "start", "persona_id": "invalid"}),
        },
        *[
            {
                "type": "websocket.receive",
                "text": json.dumps({"type": "start", "backend": backend}),
            }
            for backend in ("native", "lfm3")
        ],
        {
            "type": "websocket.receive",
            "text": json.dumps(
                {"type": "start", "persona_id": None, "instruction": "Old contract"}
            ),
        },
    ],
)
def test_malformed_start_never_resolves_persona_or_connects(monkeypatch, message):
    browser = socket()
    browser.receive.return_value = message
    connect = AsyncMock()
    monkeypatch.setattr(calls, "connect", connect)
    asyncio.run(calls.voice_call("test", browser))
    browser.app.state.services.call_instruction.assert_not_awaited()
    connect.assert_not_awaited()
    assert (
        browser.send_json.call_args.args[0]["message"] == "Invalid call start message."
    )
    browser.close.assert_awaited_once_with(1008)


def test_start_wait_is_bounded(monkeypatch):
    browser = socket()

    async def timed_out(awaitable, timeout):
        awaitable.close()
        assert timeout == 15
        raise TimeoutError

    monkeypatch.setattr(calls.asyncio, "wait_for", timed_out)
    asyncio.run(calls.voice_call("test", browser))
    assert browser.send_json.call_args.args[0]["message"] == "Call start timed out."
    browser.close.assert_awaited_once()


@pytest.mark.parametrize(
    "error, code",
    [
        (ServiceError("Tsuchi is unavailable."), 1013),
        (ServiceError("The saved persona is in trash.", 409), 1008),
        (
            ServiceError(
                "Voice call instructions exceed the 8,000-character limit.", 422
            ),
            1008,
        ),
    ],
)
def test_persona_failure_is_reported_before_hikari_connects(monkeypatch, error, code):
    browser = socket()
    browser.app.state.services.call_instruction.side_effect = error
    connect = AsyncMock()
    monkeypatch.setattr(calls, "connect", connect)
    asyncio.run(calls.voice_call("test", browser))
    assert browser.send_json.call_args.args[0] == {
        "type": "error",
        "message": str(error),
    }
    browser.close.assert_awaited_once_with(code)
    connect.assert_not_awaited()


def test_browser_disconnect_before_start_never_connects(monkeypatch):
    browser = socket()
    browser.receive.return_value = {"type": "websocket.disconnect", "code": 1000}
    connect = AsyncMock()
    monkeypatch.setattr(calls, "connect", connect)
    asyncio.run(calls.voice_call("test", browser))
    connect.assert_not_awaited()
    browser.send_json.assert_not_awaited()
    browser.close.assert_awaited_once()


def test_disconnected_browser_during_setup_still_closes_upstream(monkeypatch):
    browser = socket()
    upstream = AsyncMock()
    upstream.send.side_effect = WebSocketDisconnect()
    monkeypatch.setattr(calls, "connect", AsyncMock(return_value=upstream))
    asyncio.run(calls.voice_call("test", browser))
    upstream.close.assert_awaited_once()
    browser.close.assert_awaited_once()


@pytest.mark.parametrize("phase", ["receive", "persona", "connect", "send"])
def test_cancellation_cleans_up_each_setup_phase(monkeypatch, phase):
    async def check():
        browser = socket()
        upstream = AsyncMock()
        reached = asyncio.Event()

        async def waiting(*args, **kwargs):
            reached.set()
            await asyncio.Future()

        connect = AsyncMock(return_value=upstream)
        if phase == "receive":
            browser.receive.side_effect = waiting
        elif phase == "persona":
            browser.app.state.services.call_instruction.side_effect = waiting
        elif phase == "connect":
            connect.side_effect = waiting
        else:
            upstream.send.side_effect = waiting
        monkeypatch.setattr(calls, "connect", connect)
        task = asyncio.create_task(calls.voice_call("test", browser))
        await reached.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        browser.close.assert_awaited_once()
        if phase == "send":
            upstream.close.assert_awaited_once()
        else:
            upstream.close.assert_not_awaited()

    asyncio.run(check())


def test_failed_upstream_cleanup_still_closes_browser(monkeypatch):
    browser = socket()
    upstream = AsyncMock()
    upstream.send.side_effect = OSError("closed while starting")
    upstream.close.side_effect = OSError("closed while cleaning up")
    monkeypatch.setattr(calls, "connect", AsyncMock(return_value=upstream))
    with pytest.raises(OSError, match="closed while cleaning up"):
        asyncio.run(calls.voice_call("test", browser))
    upstream.close.assert_awaited_once()
    browser.close.assert_awaited_once_with(1013)
