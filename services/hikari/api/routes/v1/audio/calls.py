import asyncio
import json
import os
from typing import Annotated, Literal

import anyio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from api.services.inference import inference
from api.services.inference_errors import InferenceFailure
from api.services.voice_call import CallModels, VoiceCall

router = APIRouter(prefix="/v1/audio/calls", tags=["Audio"])


class CallStart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["start"]
    backend: Literal["native", "lfm3"] = "native"
    instruction: str = Field(
        default="Have a spoken conversation. Keep replies short. You have no tools.",
        min_length=1,
        max_length=8000,
    )


class CallInterrupt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["interrupt"]
    turn: int = Field(ge=1, le=2**31 - 1)


class CallUtterance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["utterance"]
    turn: int = Field(ge=1, le=2**31 - 1)
    audio: str = Field(max_length=900_000, pattern=r"^data:audio/wav;base64,.+")


class CallPlayed(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["played"]
    turn: int = Field(ge=1, le=2**31 - 1)
    segment: int = Field(ge=0, le=8000)


class CallEnd(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["end"]


call_event = TypeAdapter(
    Annotated[
        CallInterrupt | CallUtterance | CallPlayed | CallEnd,
        Field(discriminator="type"),
    ]
)


@router.websocket("")
async def voice_call(socket: WebSocket):
    await socket.accept()
    try:
        start = CallStart.model_validate(
            await asyncio.wait_for(socket.receive_json(), 15)
        )
        if start.backend == "lfm3" and os.getenv(
            "HIKARI_ENABLE_LFM3_AUDIO", ""
        ).lower() not in ("1", "true", "yes"):
            raise InferenceFailure(
                "The LFM3 audio trial is disabled on this Hikari API.", 422
            )
        models = CallModels.selected(inference.models_status(), start.backend)

        async def receive():
            return call_event.validate_python(await socket.receive_json())

        # The connection owns residency and cleanup, including cancellation of
        # transcription, streamed text and buffered Qwen segments on disconnect.
        with inference.worker.full_duplex(models.participants()):
            await VoiceCall(inference, models, start.instruction, socket.send_json).run(
                receive
            )
    except WebSocketDisconnect:
        return
    except (ValidationError, json.JSONDecodeError):
        await socket.send_json({"type": "error", "message": "Invalid call message."})
    except TimeoutError:
        await socket.send_json({"type": "error", "message": "Call start timed out."})
    except (InferenceFailure, ExceptionGroup) as error:
        message = (
            str(error)
            if isinstance(error, InferenceFailure)
            else "Voice generation failed. Start a new call."
        )
        await socket.send_json({"type": "error", "message": message})
    finally:
        with anyio.CancelScope(shield=True):
            try:
                await socket.close()
            except (RuntimeError, OSError):
                pass
