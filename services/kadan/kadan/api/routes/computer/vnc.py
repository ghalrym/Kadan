"""Relay the agent computer's VNC stream so the browser never holds its API key."""

from fastapi import APIRouter, WebSocket
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidHandshake, InvalidURI

from ...websocket_relay import relay

router = APIRouter()
# Unavailable upstream: RFC 6455 "try again later".
TRY_AGAIN_LATER = 1013


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
    await relay(socket, upstream)
