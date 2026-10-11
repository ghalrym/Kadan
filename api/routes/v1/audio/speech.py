from typing import Literal
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator
from api.pydantic_models.media import GeneratedSpeech
from api.inference.tts.catalog import SPEECH_MODELS, SPEAKERS, LANGUAGES
from api.services.inference import inference
from api.services.inference_http import infer

router = APIRouter(prefix="/v1/audio/speech", tags=["Audio"])

class CustomVoice(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    mode: Literal["custom"]
    speaker: str
    instruction: str = Field(default='', max_length=8000)



class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    script: str = Field(min_length=1)
    model_id: Literal['qwen-tts-1.7b-custom'] = 'qwen-tts-1.7b-custom'
    language: str = 'Auto'
    voice: CustomVoice

    @model_validator(mode='after')
    def validate_model(self):
        if self.language not in LANGUAGES or self.voice.speaker not in SPEAKERS:
            raise ValueError('Select a supported CustomVoice speaker and language.')
        if not 1 <= len(self.script.encode('utf-8')) <= 32000 or len(self.voice.instruction.encode('utf-8')) > 8000:
            raise ValueError('Speech text or instruction exceeds its native byte limit.')
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
    model = SPEECH_MODELS['qwen-tts-1.7b-custom']
    return [SpeechModelOption(id=model.id, name=model.name, mode='custom', speakers=list(SPEAKERS),
        supports_instruction=True, default_speaker='Ryan')]


@router.get("", operation_id="listSpeech")
def list_speech() -> SpeechHistoryResponse:
    """Return empty speech history and blank editor defaults, without fixture audio."""
    return SpeechHistoryResponse(audio=[], voice_description="", script="")


@router.post("", operation_id="generateSpeech", responses={503: {"model": SpeechUnavailable, "description": "Speech provider unavailable"}})
async def generate_speech(body: SpeechRequest, request: Request) -> SpeechResponse:
    """Queue native speech; cancellation waits for owned cleanup."""
    result = await infer(request, inference.submit(body, feature='tts'))
    return SpeechResponse(audio=result)
