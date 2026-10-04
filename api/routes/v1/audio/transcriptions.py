from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(prefix="/v1/audio/transcriptions", tags=["Audio"])

MOCK_TRANSCRIPT = ('Okay, quick update on the migration. The new GPU node is racked and passing burn-in. '
 'I would like to move batch transcription over on Thursday.')


class TranscriptionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audio: str = Field(min_length=1, description="Audio reference; not fetched in mock mode")
    formatting: bool = True


class TranscriptionResponse(BaseModel):
    text: str


@router.post("", operation_id="transcribeAudio")
def transcribe_audio(body: TranscriptionRequest) -> TranscriptionResponse:
    return TranscriptionResponse(text=MOCK_TRANSCRIPT)
