from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.services.inference_http import infer
from api.services.transcription import transcribe

router = APIRouter(prefix="/v1/audio/transcriptions", tags=["Audio"])


class TranscriptionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audio: str = Field(
        min_length=1, description="Base64 mono 16-bit PCM WAV data URL at 16000 Hz"
    )
    model: Literal["medium"] | None = None
    language: str | None = None
    formatting: bool | None = Field(
        default=None,
        description="Null or omitted uses the saved formatting setting. Enabled uses S1-mini by Superwhisper to normalize the transcript; raw_text is always retained.",
    )

    @field_validator("audio")
    @classmethod
    def validate_audio(cls, value: str) -> str:
        """Reject blank input before bounded WAV transport decoding."""
        if not value.strip():
            raise ValueError("Audio must not be blank")
        return value.strip()


class TranscriptionResponse(BaseModel):
    text: str
    raw_text: str
    language: str | None
    model: str
    formatting_status: Literal["disabled", "formatted", "failed", "empty"]
    formatting_model: str | None = None
    formatting_error: str | None = None


class TranscriptionUnavailable(BaseModel):
    detail: str


@router.post(
    "",
    operation_id="transcribeAudio",
    responses={503: {"model": TranscriptionUnavailable}},
)
async def transcribe_audio(
    body: TranscriptionRequest, request: Request
) -> TranscriptionResponse:
    """Queue native Whisper and preserve its raw transcript and formatting status."""
    if not body.audio.startswith("data:audio/wav;base64,"):
        raise HTTPException(
            422,
            "Supply a base64 PCM WAV data URL. Audio references and URLs are not fetched.",
        )
    return TranscriptionResponse(**await infer(request, transcribe(body)))
