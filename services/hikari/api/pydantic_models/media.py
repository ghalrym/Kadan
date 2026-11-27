from typing import Literal
from pydantic import BaseModel, Field


class ImageSet(BaseModel):
    id: str
    mode: Literal["Generate", "Edit"]
    prompt: str
    aspect: Literal["square", "landscape", "portrait", "wide"]
    seeds: list[int]
    meta: str
    images_base64: list[str] = Field(default_factory=list)
    mime_type: Literal["image/png"] | None = None


class GeneratedSpeech(BaseModel):
    voice: str
    meta: str
    script: str
    time: str
    audio_base64: str | None = None
    mime_type: Literal["audio/wav"] | None = None
