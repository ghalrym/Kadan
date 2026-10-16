"""Serializable jobs submitted to the inference worker's FIFO."""

from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from api.pydantic_models.chat import ChatMessage
from api.pydantic_models.decisions import DecisionQuestion
from api.pydantic_models.inference import Operation


class TaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatTask(TaskInput):
    feature: Literal["llm"] = "llm"
    messages: list[ChatMessage] = Field(default_factory=list)
    model: str | None = None
    model_id: str | None = None
    context_limit: int | None = None
    stream: bool = False
    conversation_id: str | None = None
    reuse_prefix: bool = True


# medium is GPT-OSS 120B; the ID predates the 20B checkpoint and stays stable.
LlmModel = Literal["small", "gpt-oss-20b", "medium", "large", "muse-spark-open-q3-xl"]
ImageModel = Literal["qwen-image-2.1", "qwen-image-2.1-turbo"]


class ImageTask(TaskInput):
    feature: Literal["image"] = "image"
    model_id: ImageModel | None = None
    prompt: str
    aspect: Literal["1:1", "4:3", "3:4", "16:9"] = "1:1"
    count: Literal[1, 2, 4] = 4
    seed: int | None = None
    steps: int = 50
    image: str | None = None
    mask: str | None = None
    strength: float = Field(default=0.65, ge=0, le=1)

    @model_validator(mode="after")
    def mask_requires_image(self):
        if self.mask is not None and self.image is None:
            raise ValueError("An image mask requires a source image")
        return self


class DecisionTask(TaskInput):
    feature: Literal["decisions"] = "decisions"
    state: str
    questions: list[DecisionQuestion]


class SpeechVoice(TaskInput):
    mode: Literal["custom"]
    speaker: str
    instruction: str = ""


class SpeechTask(TaskInput):
    feature: Literal["tts"] = "tts"
    script: str
    model_id: str = "qwen-tts-1.7b-custom"
    language: str = "Auto"
    voice: SpeechVoice


class TranscriptionTask(TaskInput):
    feature: Literal["stt"] = "stt"
    audio: str = Field(max_length=2 * 1024**2)
    model: str | None = None
    language: str | None = None
    formatting: bool = True


class VideoTask(TaskInput):
    feature: Literal["video"] = "video"
    model: str = "h3-fl2va-int8-turbo"
    seed: int = 42
    prompt: str
    negative_prompt: Literal[""] = ""
    duration: int = 8
    fps: Literal[24] = 24
    resolution: Literal["480p", "768p"] = "768p"
    aspect: Literal["16:9", "9:16", "1:1"] = "16:9"


class DownloadTask(TaskInput):
    feature: Literal["models"] = "models"
    model_id: str
    license_acknowledged: bool = False


class ModelSelectionTask(TaskInput):
    feature: Literal["model_selection"] = "model_selection"
    model_id: str


class ModelContextTask(TaskInput):
    feature: Literal["model_context"] = "model_context"
    model_id: str
    context_limit: int | None = Field(ge=1, le=2**31 - 1)


class WhisperSelectionTask(TaskInput):
    feature: Literal["whisper_selection"] = "whisper_selection"
    model: str


class ImageSelectionTask(TaskInput):
    feature: Literal["image_selection"] = "image_selection"
    model_id: ImageModel


InferenceInput = Annotated[
    ChatTask
    | ImageTask
    | DecisionTask
    | SpeechTask
    | TranscriptionTask
    | VideoTask
    | DownloadTask
    | ModelSelectionTask
    | ModelContextTask
    | WhisperSelectionTask
    | ImageSelectionTask,
    Field(discriminator="feature"),
]


class InferenceTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(default_factory=lambda: uuid4().hex, pattern=r"^[a-f0-9]{32}$")
    command: Literal["submit"] = "submit"
    operation: Operation = "generate"
    input: InferenceInput

    @field_serializer("input")
    def serialize_input(self, value):
        return {
            **value.model_dump(mode="json", exclude_unset=True),
            "feature": value.feature,
        }

    @property
    def feature(self):
        return self.input.feature

    @model_validator(mode="after")
    def validate_operation(self):
        settings = self.feature in (
            "model_selection",
            "model_context",
            "whisper_selection",
            "image_selection",
        )
        if settings != (self.operation == "configure"):
            raise ValueError("Model settings require the configure operation")
        if (self.feature == "models") != (self.operation == "download"):
            raise ValueError("Model downloads require the download operation")
        if self.operation in ("load", "unload", "completion") and self.feature != "llm":
            raise ValueError("Only language tasks support this operation")
        if self.operation == "completion" and not self.input.messages:
            raise ValueError("A completion requires messages")
        return self
