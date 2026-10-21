import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from kadan.api.websocket_relay import relay


def test_disconnect_joins_both_pumps_and_closes_upstream():
    async def check():
        entered, cancelled = asyncio.Event(), asyncio.Event()

        async def receive():
            await entered.wait()
            return {"type": "websocket.disconnect"}

        class Upstream:
            send = AsyncMock()
            close = AsyncMock()

            async def __aiter__(self):
                entered.set()
                try:
                    await asyncio.Future()
                    yield "unreachable"
                finally:
                    cancelled.set()

        browser = SimpleNamespace(receive=receive, close=AsyncMock())
        upstream = Upstream()
        await relay(browser, upstream)
        assert cancelled.is_set()
        upstream.close.assert_awaited_once()
        browser.close.assert_awaited_once()

    asyncio.run(check())


def test_text_relay_preserves_turn_messages_in_both_directions():
    async def check():
        sent = []
        incoming = iter(
            (
                {"type": "websocket.receive", "text": '{"type":"interrupt","turn":2}'},
                {"type": "websocket.disconnect"},
            )
        )

        class Upstream:
            send = AsyncMock()
            close = AsyncMock()

            async def __aiter__(self):
                yield '{"type":"ready"}'
                await asyncio.Future()

        async def receive():
            while not sent:
                await asyncio.sleep(0)
            return next(incoming)

        async def send_text(text):
            sent.append(text)

        browser = SimpleNamespace(
            receive=receive, send_text=send_text, close=AsyncMock()
        )
        upstream = Upstream()
        await relay(browser, upstream)
        upstream.send.assert_awaited_once_with('{"type":"interrupt","turn":2}')
        assert sent == ['{"type":"ready"}']

    asyncio.run(check())
