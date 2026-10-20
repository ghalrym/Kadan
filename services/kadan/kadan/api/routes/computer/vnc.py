"""Relay the agent computer's VNC stream so the browser never holds its API key."""

import asyncio
import logging

from fastapi import APIRouter, WebSocket
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidURI

router = APIRouter()
logger = logging.getLogger(__name__)
# Unavailable upstream: RFC 6455 "try again later".
TRY_AGAIN_LATER = 1013


async def browser_to_computer(socket: WebSocket, upstream: ClientConnection) -> None:
    while True:
        message = await socket.receive()
        if message["type"] == "websocket.disconnect":
            return
        data = message.get("bytes")
        await upstream.send(data if data is not None else message.get("text", ""))


async def computer_to_browser(socket: WebSocket, upstream: ClientConnection) -> None:
    async for data in upstream:
        if isinstance(data, bytes):
            await socket.send_bytes(data)
        else:
            await socket.send_text(data)


@router.websocket("/api/computer/vnc")
async def computer_vnc(socket: WebSocket) -> None:
    sandbox = socket.app.state.services.sandbox
    await socket.accept()
    if not sandbox.configured:
        await socket.close(TRY_AGAIN_LATER, "No computer is configured.")
        return
    try:
        upstream = await connect(
            sandbox.vnc_url(),
            additional_headers=sandbox.headers(),
            open_timeout=5,
            max_size=None,
            proxy=None,
        )
    except (OSError, TimeoutError, InvalidHandshake, InvalidURI):
        await socket.close(TRY_AGAIN_LATER, "The agent's computer is unavailable.")
        return
    tasks = {
        asyncio.create_task(browser_to_computer(socket, upstream)),
        asyncio.create_task(computer_to_browser(socket, upstream)),
    }
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            error = task.exception()
            if error is not None and not isinstance(error, ConnectionClosed):
                logger.info("Computer VNC relay ended: %s", error)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await upstream.close()
        try:
            await socket.close()
        except RuntimeError:
            # The browser already closed its side.
            pass
