from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

router = APIRouter(prefix="/v1/audio/transcriptions", tags=["Audio"])


class TranscriptionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audio: str = Field(min_length=1, max_length=2048, description="Audio reference only; no upload or fetching is implemented")
    formatting: bool = True

    @field_validator("audio")
    @classmethod
    def validate_reference(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Audio reference must not be blank")
        return value


class TranscriptionResponse(BaseModel):
    text: str


class TranscriptionUnavailable(BaseModel):
    detail: str


@router.post("", operation_id="transcribeAudio", responses={503: {"model": TranscriptionUnavailable}})
def transcribe_audio(body: TranscriptionRequest) -> TranscriptionResponse:
    raise HTTPException(
        status_code=503,
        detail="Transcription is unavailable: no speech-to-text provider is configured. Audio references are not fetched; recording and file upload are not supported yet.",
    )
