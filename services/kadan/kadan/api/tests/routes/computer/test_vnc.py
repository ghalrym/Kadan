import asyncio
from types import SimpleNamespace

from kadan.api.routes.computer import vnc


class Socket:
    def __init__(self, sandbox, browser):
        self.app = SimpleNamespace(
            state=SimpleNamespace(services=SimpleNamespace(sandbox=sandbox))
        )
        self.browser = list(browser)
        self.sent = []
        self.closed = None

    async def accept(self):
        pass

    async def receive(self):
        return self.browser.pop(0)

    async def send_bytes(self, data):
        self.sent.append(data)

    async def send_text(self, data):
        self.sent.append(data)

    async def close(self, code=1000, reason=None):
        self.closed = (code, reason)


class Upstream:
    def __init__(self, frames):
        self.frames = frames
        self.received = []
        self.closed = False

    async def send(self, data):
        self.received.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.frames:
            await asyncio.sleep(10)
        return self.frames.pop(0)

    async def close(self):
        self.closed = True


def sandbox():
    return SimpleNamespace(
        configured=True,
        vnc_url=lambda: "ws://kadan-sandbox:8080/websockify",
        headers=lambda: {"Authorization": "Bearer secret"},
    )


def test_unconfigured_computer_closes_the_browser_socket():
    socket = Socket(SimpleNamespace(configured=False), [])
    asyncio.run(vnc.computer_vnc(socket))
    assert socket.closed[0] == vnc.TRY_AGAIN_LATER


def test_unreachable_computer_closes_the_browser_socket(monkeypatch):
    async def refuse(*args, **kwargs):
        raise OSError("refused")

    monkeypatch.setattr(vnc, "connect", refuse)
    socket = Socket(sandbox(), [])
    asyncio.run(vnc.computer_vnc(socket))
    assert socket.closed == (
        vnc.TRY_AGAIN_LATER,
        "The agent's computer is unavailable.",
    )


def test_relay_carries_frames_both_ways_with_the_key_kept_server_side(monkeypatch):
    upstream = Upstream([b"RFB 003.008\n"])
    connections = []

    async def connect(url, **kwargs):
        connections.append((url, kwargs["additional_headers"]))
        return upstream

    monkeypatch.setattr(vnc, "connect", connect)

    class Browser(Socket):
        async def receive(self):
            # Wait for the server greeting before answering, then leave.
            while not self.sent:
                await asyncio.sleep(0)
            return self.browser.pop(0)

    socket = Browser(
        sandbox(),
        [
            {"type": "websocket.receive", "bytes": b"RFB 003.008\n"},
            {"type": "websocket.disconnect"},
        ],
    )
    asyncio.run(vnc.computer_vnc(socket))
    assert connections == [
        ("ws://kadan-sandbox:8080/websockify", {"Authorization": "Bearer secret"})
    ]
    assert socket.sent == [b"RFB 003.008\n"]
    assert upstream.received == [b"RFB 003.008\n"]
    assert upstream.closed
