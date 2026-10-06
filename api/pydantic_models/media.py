from typing import Literal
from pydantic import BaseModel, Field


class ImageSet(BaseModel):
    id: str
    mode: Literal["Generate", "Edit"]
    prompt: str
    aspect: Literal["square", "landscape", "portrait", "wide"]
    seeds: list[int]
    meta: str


class VideoJob(BaseModel):
    id: str
    prompt: str
    duration: str
    resolution: str
    aspect: Literal["wide", "portrait", "square"]
    fps: str
    progress: int = Field(ge=0, le=100)
    time: str
    status: Literal["Rendering", "Queued", "Done", "Failed", "Cancelled"]
    thumbnail: str
    output_url: str | None = None
    error: str | None = None
    progress_text: str = Field(alias="progressText")


class GeneratedSpeech(BaseModel):
    voice: str
    meta: str
    script: str
    time: str
    audio_base64: str | None = None
    mime_type: Literal["audio/wav"] | None = None
