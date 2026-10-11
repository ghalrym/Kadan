"""The API-to-native request data contract."""
from typing import Literal

from pydantic import BaseModel, ConfigDict

from api.pydantic_models.chat import ChatMessage
from api.pydantic_models.decisions import DecisionQuestion

Feature = Literal['llm', 'image', 'video', 'stt', 'tts', 'decisions']
Operation = Literal['generate', 'completion', 'load', 'unload']


class ChatInput(BaseModel):
    messages: list[ChatMessage]
    max_output_tokens: int = 256


class DecisionInput(BaseModel):
    state: str
    questions: list[DecisionQuestion]


class ImageInput(BaseModel):
    prompt: str
    width: int
    height: int
    steps: int
    seed: int
    count: int


class SpeechInput(BaseModel):
    script: str
    speaker: str
    language: str
    instruction: str


class VideoInput(BaseModel):
    prompt: str
    short_edge: int
    aspect: str
    duration: int
    seed: int


class VideoPaths(BaseModel):
    tokenizer: str
    text: str
    denoiser: str
    turbo: str
    vae: str
    audio_vae: str


class Payload(BaseModel):
    model_config = ConfigDict(extra='forbid')


class EmptyPayload(Payload):
    """Unload or cancelled preparation acknowledgement."""


class ChatPayload(Payload):
    checkpoint: str
    architecture: str
    context_limit: int
    input: str | None = None


class DecisionPayload(Payload):
    checkpoint: str
    request: DecisionInput


class ImagePayload(Payload):
    checkpoint: str
    output: str
    request: ImageInput


class SpeechPayload(Payload):
    checkpoint: str
    tokenizer: str
    output: str
    request: SpeechInput


class TranscriptionPayload(Payload):
    checkpoint: str
    assets: str
    input: str


class VideoPayload(Payload):
    paths: VideoPaths
    output: str
    request: VideoInput


NativePayload = EmptyPayload | ChatPayload | DecisionPayload | ImagePayload | SpeechPayload | TranscriptionPayload | VideoPayload
