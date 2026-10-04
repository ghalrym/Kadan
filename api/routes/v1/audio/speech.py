from typing import Annotated, Literal
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field
from api.pydantic_models.media import GeneratedSpeech

router = APIRouter(prefix="/v1/audio/speech", tags=["Audio"])

MOCK_SPEECH = GeneratedSpeech(
    voice='Described · warm, low, calm',
    meta='F5-TTS · 3.6 s · WAV · 14:11',
    script='Your order has shipped and will arrive on Thursday.',
    time='0:03',
)

MOCK_VOICE_DESCRIPTION = 'Warm, low female voice, mid-40s, calm and unhurried'
MOCK_SCRIPT = ('Welcome back. Your server has been up for twelve days, and everything is running '
 'normally.')


class DescribedVoice(BaseModel):
    mode: Literal["describe"]
    description: str = Field(min_length=1)


class ClonedVoice(BaseModel):
    mode: Literal["clone"]
    sample: str = Field(min_length=1, description="Voice sample reference; not fetched in mock mode")


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    script: str = Field(min_length=1, max_length=5000)
    voice: Annotated[DescribedVoice | ClonedVoice, Field(discriminator="mode")]


class SpeechResponse(BaseModel):
    audio: GeneratedSpeech


class SpeechHistoryResponse(BaseModel):
    audio: list[GeneratedSpeech]
    voice_description: str
    script: str


@router.get("")
def list_speech() -> SpeechHistoryResponse:
    return SpeechHistoryResponse(audio=[MOCK_SPEECH], voice_description=MOCK_VOICE_DESCRIPTION, script=MOCK_SCRIPT)


@router.post("")
def generate_speech(body: SpeechRequest) -> SpeechResponse:
    return SpeechResponse(audio=MOCK_SPEECH)
