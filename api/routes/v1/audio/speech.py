from dataclasses import asdict
from typing import Annotated, Literal
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator
from api.pydantic_models.media import GeneratedSpeech
from api.services.speech import validate_request, speech_models
from api.memory_manager import memory_manager
from api.memory_manager.http import infer

router = APIRouter(prefix="/v1/audio/speech", tags=["Audio"])

class DescribedVoice(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    mode: Literal["describe"]
    description: str = Field(min_length=1)


class ClonedVoice(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    mode: Literal["clone"]
    sample: str = Field(min_length=1, description="Base64-encoded audio bytes; URLs and server paths are not accepted.")
    transcript: str | None = None
    speaker_only: bool = False


class CustomVoice(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    mode: Literal["custom"]
    speaker: str
    instruction: str = ''



class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    script: str = Field(min_length=1)
    model_id: str | None = None
    language: str = 'Auto'
    voice: Annotated[DescribedVoice | ClonedVoice | CustomVoice, Field(discriminator="mode")]

    @model_validator(mode='after')
    def validate_model(self):
        """Adapter validation runs before allocation; the route has no model rules."""
        self.model_id = validate_request(self.model_dump())
        return self


class SpeechResponse(BaseModel):
    audio: GeneratedSpeech


class SpeechHistoryResponse(BaseModel):
    audio: list[GeneratedSpeech]
    voice_description: str
    script: str


class SpeechUnavailable(BaseModel):
    detail: str


class SpeechModelOption(BaseModel):
    id: str
    name: str
    mode: Literal['custom', 'describe', 'clone']
    speakers: list[str] = Field(default_factory=list)
    supports_instruction: bool = False
    default_speaker: str | None = None


@router.get('/models', operation_id='listSpeechModels')
def list_speech_models() -> list[SpeechModelOption]:
    """Expose only enabled native checkpoint integrations."""
    return [SpeechModelOption(**asdict(item)) for item in speech_models()]


@router.get("", operation_id="listSpeech")
def list_speech() -> SpeechHistoryResponse:
    """Return empty speech history and blank editor defaults, without fixture audio."""
    return SpeechHistoryResponse(audio=[], voice_description="", script="")


@router.post("", operation_id="generateSpeech", responses={503: {"model": SpeechUnavailable, "description": "Speech provider unavailable"}})
async def generate_speech(body: SpeechRequest, request: Request) -> SpeechResponse:
    """Queue native speech; cancellation waits for owned cleanup."""
    result = await infer(request, memory_manager.submit(body, feature='tts'))
    return SpeechResponse(audio=result)
