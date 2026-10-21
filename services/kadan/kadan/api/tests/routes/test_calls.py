import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from kadan.api.routes import calls


def socket(url="http://hikari-api:8000"):
    value = AsyncMock()
    value.app = SimpleNamespace(
        state=SimpleNamespace(
            services=SimpleNamespace(settings=SimpleNamespace(hikari_url=url))
        )
    )
    return value


def test_call_reuses_local_hikari_url_without_credentials_or_a_new_service(monkeypatch):
    browser = socket("https://hikari.local/prefix")
    upstream = object()
    connect = AsyncMock(return_value=upstream)
    relay = AsyncMock()
    monkeypatch.setattr(calls, "connect", connect)
    monkeypatch.setattr(calls, "relay", relay)
    asyncio.run(calls.voice_call(browser))
    connect.assert_awaited_once_with(
        "wss://hikari.local/prefix/v1/audio/calls",
        open_timeout=5,
        max_size=6 * 1024**2,
        proxy=None,
    )
    relay.assert_awaited_once_with(browser, upstream)


def test_unavailable_hikari_closes_call_with_actionable_error(monkeypatch):
    browser = socket()
    monkeypatch.setattr(calls, "connect", AsyncMock(side_effect=OSError("refused")))
    asyncio.run(calls.voice_call(browser))
    assert browser.send_json.call_args.args[0]["type"] == "error"
    browser.close.assert_awaited_once_with(1013)
