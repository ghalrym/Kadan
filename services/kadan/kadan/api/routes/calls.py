import asyncio
import json
from contextlib import suppress
from typing import Literal
from uuid import UUID

import anyio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, ValidationError
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidURI

from ..config import ServiceError
from ..pydantic_models.resources import AgentKey
from ..websocket_relay import relay

router = APIRouter()


class CallStart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["start"]
    persona_id: UUID | None


@router.websocket("/api/agents/{key}/calls")
async def voice_call(key: AgentKey, socket: WebSocket):
    await socket.accept()
    upstream = None
    relaying = False
    close_code = 1008
    try:
        message = await asyncio.wait_for(socket.receive(), 15)
        if message["type"] == "websocket.disconnect":
            return
        start = CallStart.model_validate_json(message.get("text") or "")
        services = socket.app.state.services
        instruction = await services.call_instruction(key, start.persona_id)
        url = services.settings.hikari_url
        url = (
            ("wss" if url.startswith("https:") else "ws")
            + url[url.index(":") :]
            + "/v1/audio/calls"
        )
        try:
            upstream = await connect(
                url, open_timeout=5, max_size=6 * 1024**2, proxy=None
            )
            await upstream.send(
                json.dumps({"type": "start", "instruction": instruction})
            )
        except (
            OSError,
            TimeoutError,
            ConnectionClosed,
            InvalidHandshake,
            InvalidURI,
        ) as error:
            raise ServiceError(
                "Hikari is unavailable. Check its local service and try again."
            ) from error
        relaying = True
        await relay(socket, upstream)
    except WebSocketDisconnect:
        return
    except ValidationError:
        await socket.send_json(
            {"type": "error", "message": "Invalid call start message."}
        )
    except TimeoutError:
        await socket.send_json({"type": "error", "message": "Call start timed out."})
    except ServiceError as error:
        close_code = 1013 if error.status >= 500 else 1008
        await socket.send_json({"type": "error", "message": str(error)})
    finally:
        # The relay owns cleanup once started; setup owns all earlier failures.
        if not relaying:
            with anyio.CancelScope(shield=True):
                try:
                    if upstream is not None:
                        await upstream.close()
                finally:
                    with suppress(RuntimeError, OSError):
                        await socket.close(close_code)
