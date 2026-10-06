from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.memory_manager import memory_manager
from api.memory_manager.http import infer
from api.services.transcription.transcription import get_transcription_manager
from api.services.transcription.whisper_catalog import get_whisper_checkpoints, checkpoint

router = APIRouter(prefix="/v1/audio/transcriptions", tags=["Audio"])


class TranscriptionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audio: str = Field(min_length=1, description="Base64 mono 16-bit PCM WAV data URL at 16000 Hz")
    model: str | None = None
    language: str | None = None
    formatting: bool = True

    @field_validator("audio")
    @classmethod
    def validate_audio(cls, value: str) -> str:
        """Reject blank input; decoding occurs under the native memory lease."""
        if not value.strip():
            raise ValueError("Audio must not be blank")
        return value.strip()

    @field_validator('model')
    @classmethod
    def validate_model(cls, value):
        return checkpoint(value).name if value is not None else None


class TranscriptionResponse(BaseModel):
    text: str
    raw_text: str
    language: str | None
    model: str
    formatting_status: str
    formatting_model: str | None = None


class TranscriptionUnavailable(BaseModel):
    detail: str


class WhisperModels(BaseModel):
    models: list[str]
    selected: str | None


class WhisperSelection(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str

    @field_validator('model')
    @classmethod
    def canonical_model(cls, value):
        return checkpoint(value).name


@router.get('/models', operation_id='getWhisperModels')
def get_models() -> WhisperModels:
    """List all unique official checkpoints and the persisted selection."""
    return WhisperModels(models=list(get_whisper_checkpoints()), selected=get_transcription_manager().selected())


@router.put('/models', operation_id='selectWhisperModel')
def select_model(body: WhisperSelection) -> WhisperModels:
    try:
        get_transcription_manager().select(body.model)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(503, 'Cannot persist Whisper selection.') from exc
    return get_models()


@router.post("", operation_id="transcribeAudio", responses={503: {"model": TranscriptionUnavailable}})
async def transcribe_audio(body: TranscriptionRequest, request: Request) -> TranscriptionResponse:
    """Queue native Whisper and preserve its raw transcript and formatting status."""
    if not body.audio.startswith('data:audio/wav;base64,'):
        raise HTTPException(422, 'Supply a base64 PCM WAV data URL. Audio references and URLs are not fetched.')
    return TranscriptionResponse(**await infer(request, memory_manager.stt(body)))
