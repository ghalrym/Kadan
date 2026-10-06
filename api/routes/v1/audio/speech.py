from typing import Annotated, Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from api.pydantic_models.media import GeneratedSpeech

router = APIRouter(prefix="/v1/audio/speech", tags=["Audio"])

class DescribedVoice(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    mode: Literal["describe"]
    description: str = Field(min_length=1)


class ClonedVoice(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    mode: Literal["clone"]
    sample: str = Field(min_length=1, description="Opaque sample reference. No upload endpoint or speech provider is configured; the reference is not fetched.")


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    script: str = Field(min_length=1)
    voice: Annotated[DescribedVoice | ClonedVoice, Field(discriminator="mode")]


class SpeechResponse(BaseModel):
    audio: GeneratedSpeech


class SpeechHistoryResponse(BaseModel):
    audio: list[GeneratedSpeech]
    voice_description: str
    script: str


class SpeechUnavailable(BaseModel):
    detail: str


@router.get("", operation_id="listSpeech")
def list_speech() -> SpeechHistoryResponse:
    """Return empty speech history and blank editor defaults, without fixture audio."""
    return SpeechHistoryResponse(audio=[], voice_description="", script="")


@router.post("", operation_id="generateSpeech", responses={503: {"model": SpeechUnavailable, "description": "No speech provider is configured"}})
def generate_speech(body: SpeechRequest) -> SpeechResponse:
    """Reject a validated describe/clone request with HTTP 503.

    No speech provider is configured, so no audio is generated or persisted and
    clone sample references are neither fetched nor treated as uploaded files."""
    raise HTTPException(status_code=503, detail="No speech provider is configured. Speech generation and voice cloning are unavailable.")
