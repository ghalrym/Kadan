import asyncio
import base64
import binascii
import threading
from typing import Annotated, Literal
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator
from api.pydantic_models.media import GeneratedSpeech
from api.services.speech import generate_speech as synthesize, SpeechUnavailable as ProviderUnavailable
from api.services.qwen_tts_catalog import SPEECH_MODELS, SPEAKERS, LANGUAGES
from api.services.speech_enabled import ENABLED_SPEECH_MODELS
from api.inference.resources import ResourceBusy, ResourceCancelled, ResourceExhausted

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
        """Reject unsupported model/mode combinations before any worker allocation."""
        defaults = {'describe': 'qwen-tts-1.7b-design', 'clone': 'qwen-tts-1.7b-base', 'custom': 'qwen-tts-1.7b-custom'}
        self.model_id = self.model_id or defaults[self.voice.mode]
        if self.model_id not in SPEECH_MODELS or SPEECH_MODELS[self.model_id].mode != self.voice.mode:
            raise ValueError('Choose a Qwen3-TTS model supporting this voice mode')
        if self.language not in LANGUAGES:
            raise ValueError('Unsupported speech language')
        if self.voice.mode == 'custom':
            if self.voice.speaker not in SPEAKERS:
                raise ValueError('Unsupported speaker')
            if self.model_id == 'qwen-tts-0.6b-custom' and self.voice.instruction:
                raise ValueError('Instruction control requires the 1.7B CustomVoice model')
        if self.voice.mode == 'clone':
            if not self.voice.speaker_only and not (self.voice.transcript or '').strip():
                raise ValueError('Provide the reference transcript or select speaker-only cloning')
            try:
                if not base64.b64decode(self.voice.sample, validate=True):
                    raise ValueError('Empty reference audio')
            except binascii.Error as exc:
                raise ValueError('Reference audio must be base64-encoded audio bytes') from exc
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


@router.get('/models', operation_id='listSpeechModels')
def list_speech_models() -> list[SpeechModelOption]:
    """Expose only enabled native checkpoint integrations."""
    return [SpeechModelOption(id=item.id, name=item.name, mode=item.mode)
            for item in SPEECH_MODELS.values() if item.id in ENABLED_SPEECH_MODELS]


@router.get("", operation_id="listSpeech")
def list_speech() -> SpeechHistoryResponse:
    """Return empty speech history and blank editor defaults, without fixture audio."""
    return SpeechHistoryResponse(audio=[], voice_description="", script="")


@router.post("", operation_id="generateSpeech", responses={503: {"model": SpeechUnavailable, "description": "Speech worker unavailable"}})
async def generate_speech(body: SpeechRequest, request: Request) -> SpeechResponse:
    """Return complete WAV audio; disconnects cancel and reap the owned worker."""
    cancel = threading.Event()
    task = asyncio.create_task(asyncio.to_thread(synthesize, body.model_dump(), cancel))
    try:
        while not task.done():
            if await request.is_disconnected():
                cancel.set()
            await asyncio.sleep(.05)
        return SpeechResponse(audio=await task)
    except (ProviderUnavailable, ResourceExhausted) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ResourceBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ResourceCancelled as exc:
        raise HTTPException(status_code=499, detail=str(exc)) from exc
    finally:
        cancel.set()
        if not task.done():
            await asyncio.shield(task)
