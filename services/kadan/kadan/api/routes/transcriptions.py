from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter()


class TranscriptionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Twenty seconds of mono 16-kHz PCM, including base64 and the WAV header.
    audio: str = Field(max_length=900_000, pattern=r"^data:audio/wav;base64,.+")


class TranscriptionResult(BaseModel):
    text: str = Field(max_length=16_000)


@router.post("/api/transcriptions")
async def transcribe(
    value: TranscriptionInput, request: Request
) -> TranscriptionResult:
    result = await request.app.state.services.json_request(
        "Hikari",
        "POST",
        "/v1/audio/transcriptions",
        json={"audio": value.audio, "formatting": False},
        timeout=120,
    )
    return TranscriptionResult.model_validate(result)
