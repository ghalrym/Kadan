from fastapi import APIRouter, WebSocket
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidHandshake, InvalidURI

from ..websocket_relay import relay

router = APIRouter()


@router.websocket("/api/calls")
async def voice_call(socket: WebSocket):
    await socket.accept()
    url = socket.app.state.services.settings.hikari_url
    url = (
        ("wss" if url.startswith("https:") else "ws")
        + url[url.index(":") :]
        + "/v1/audio/calls"
    )
    try:
        upstream = await connect(url, open_timeout=5, max_size=6 * 1024**2, proxy=None)
    except (OSError, TimeoutError, InvalidHandshake, InvalidURI):
        await socket.send_json(
            {
                "type": "error",
                "message": "Hikari is unavailable. Check its local service and try again.",
            }
        )
        await socket.close(1013)
        return
    await relay(socket, upstream)
