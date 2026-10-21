"""Relay existing local WebSocket services with connection-owned cleanup."""

import asyncio
from contextlib import suppress

import anyio
from fastapi import WebSocket
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed


async def relay(socket: WebSocket, upstream: ClientConnection):
    async def upload():
        while True:
            message = await socket.receive()
            if message["type"] == "websocket.disconnect":
                return
            data = message.get("bytes")
            await upstream.send(data if data is not None else message.get("text", ""))

    async def download():
        async for data in upstream:
            if isinstance(data, bytes):
                await socket.send_bytes(data)
            else:
                await socket.send_text(data)

    tasks = [asyncio.create_task(upload()), asyncio.create_task(download())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            with suppress(ConnectionClosed):
                await task
    finally:
        with anyio.CancelScope(shield=True):
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await upstream.close()
            with suppress(RuntimeError, OSError):
                await socket.close()
