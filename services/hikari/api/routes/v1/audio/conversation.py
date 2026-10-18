import json

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from api.pydantic_models.chat import ChatMessage
from api.pydantic_models.model_store import TierId
from api.routes.v1.audio.speech import SPEAKERS
from api.services import voice_conversation
from api.services.inference_errors import InferenceFailure
from api.services.inference_http import OwnedStreamResponse, infer

router = APIRouter(prefix="/v1/audio/conversation", tags=["Audio"])


class SpeechBackendOption(BaseModel):
    id: voice_conversation.SpeechBackend
    name: str
    speech_model: str | None
    available: bool
    reason: str | None
    streaming: bool


class ConversationCapabilities(BaseModel):
    available: bool
    reason: str | None
    tier: TierId | None
    asr_model: str | None
    chat_model: str | None
    scheduling: str
    backends: list[SpeechBackendOption]
    notes: list[str]


class WindowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tier: TierId
    model: str
    audio: str = Field(
        min_length=1, max_length=2 * 1024**2, pattern=r"^data:audio/wav;base64,"
    )


class WindowResponse(BaseModel):
    text: str
    model: str


class VoiceMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    content: str = Field(min_length=1, max_length=8000)


class TurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tier: TierId
    backend: voice_conversation.SpeechBackend
    messages: list[VoiceMessage] = Field(min_length=1, max_length=40)
    speaker: str = "Ryan"

    @model_validator(mode="after")
    def bounded_turn(self):
        if self.speaker not in SPEAKERS:
            raise ValueError("Choose a supported Qwen speaker.")
        if any(
            message.role not in ("system", "user", "assistant")
            for message in self.messages
        ):
            raise ValueError("Invalid conversation role.")
        if (
            sum(len(message.content.encode("utf-8")) for message in self.messages)
            > 32000
        ):
            raise ValueError(
                "Conversation history exceeds 32000 bytes. Start a new session."
            )
        return self


@router.get("/capabilities", operation_id="getConversationCapabilities")
def get_capabilities() -> ConversationCapabilities:
    return ConversationCapabilities(**voice_conversation.capabilities())


@router.post("/transcriptions", operation_id="transcribeConversationWindow")
async def transcribe_window(body: WindowRequest, request: Request) -> WindowResponse:
    return WindowResponse(
        **await infer(
            request,
            voice_conversation.transcribe_window(
                body.audio,
                body.tier,
                body.model,
            ),
        )
    )


async def turn_events(body: TurnRequest):
    stream = voice_conversation.conversation_turn(
        body.tier,
        body.backend,
        [
            ChatMessage(role=message.role, text=message.content)
            for message in body.messages
        ],
        body.speaker,
    )
    try:
        async for event in stream:
            yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
    except InferenceFailure as exc:
        yield "data: " + json.dumps({"type": "error", "detail": str(exc)}) + "\n\n"
    finally:
        await stream.aclose()


@router.post(
    "/turn",
    operation_id="streamConversationTurn",
    responses={
        200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}},
    },
)
async def stream_turn(body: TurnRequest):
    try:
        voice_conversation.require_backend(
            voice_conversation.inference, body.tier, body.backend
        )
    except InferenceFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    events = turn_events(body)
    return OwnedStreamResponse(events, events)
