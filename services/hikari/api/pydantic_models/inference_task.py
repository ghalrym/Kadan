"""Serializable jobs submitted to the inference worker's FIFO."""

from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from api.pydantic_models.chat import ChatMessage, Lfm3Message, Lfm3Tool, lfm3_messages, require_lfm3_tools
from api.pydantic_models.decisions import DecisionQuestion
from api.pydantic_models.inference import Operation
from api.pydantic_models.model_store import TierId


class TaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatTask(TaskInput):
    feature: Literal["llm"] = "llm"
    messages: list[ChatMessage | Lfm3Message] = Field(default_factory=list)
    tools: list[Lfm3Tool] | None = Field(default=None, max_length=256)
    model: str | None = None
    model_id: str | None = None
    context_limit: int | None = None
    stream: bool = False
    conversation_id: str | None = None
    reuse_prefix: bool = True

    @model_validator(mode="before")
    @classmethod
    def experimental_messages(cls, values):
        return lfm3_messages(values)

    @model_validator(mode="after")
    def experimental_tools_only(self):
        require_lfm3_tools(self.model_id or self.model, self.messages, self.tools)
        return self


# One GGUF mixture-of-experts language model per hardware tier.
LlmModel = Literal[
    "qwen3.6-35b-a3b-q4-k-m",
    "qwen3.8-flash-next-ud-iq3-xxs",
    "qwen3.8-flash-next-q3-k-xl",
]
# Every hardware tier generates images with Turbo.
ImageModel = Literal["qwen-image-2.1-turbo"]


class ImageTask(TaskInput):
    feature: Literal["image"] = "image"
    model_id: ImageModel | None = None
    prompt: str
    aspect: Literal["1:1", "4:3", "3:4", "16:9"] = "1:1"
    count: Literal[1, 2, 4] = 4
    batch_collection_target: int | None = Field(
        default=None,
        ge=1,
        le=32,
        strict=True,
        description="Expected image count in the submitting cohort. Only shortens collection waiting; queued compatible requests may still share a larger admitted batch.",
    )
    seed: int | None = None
    steps: int = 50
    image: str | None = None
    mask: str | None = None
    strength: float = Field(default=0.65, ge=0, le=1)
    mode: Literal["source", "reference"] = "source"
    width: int | None = Field(default=None, ge=32, le=3072, multiple_of=32)
    height: int | None = Field(default=None, ge=32, le=3072, multiple_of=32)

    @model_validator(mode="after")
    def validate_edit_mode(self):
        if self.mask is not None and self.image is None:
            raise ValueError("An image mask requires a source image")
        if (self.width is None) != (self.height is None):
            raise ValueError(
                "Reference output width and height must be supplied together"
            )
        if self.mode == "reference":
            if self.image is None:
                raise ValueError("Reference editing requires an image")
            if self.mask is not None or "strength" in self.model_fields_set:
                raise ValueError("Reference editing does not accept a mask or strength")
        elif self.width is not None:
            raise ValueError("Output dimensions can only be set for reference editing")
        if self.width is not None and self.width * self.height > 4_718_592:
            raise ValueError("Reference output may contain at most 4,718,592 pixels")
        return self


class DecisionTask(TaskInput):
    feature: Literal["decisions"] = "decisions"
    state: str
    questions: list[DecisionQuestion]


class VisionTask(TaskInput):
    feature: Literal["vision"] = "vision"
    model: Literal["lfm3-vl-2b-bf16"] = "lfm3-vl-2b-bf16"
    image_base64: str = Field(min_length=1, max_length=2_796_204)
    media_type: Literal["image/png", "image/jpeg"]
    question: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=1024, ge=1, le=1024, strict=True)


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
    formatting: bool | None = None


class AudioCallTask(TaskInput):
    """Bounded buffered turn on the explicitly selected experimental audio worker."""

    feature: Literal["audio_call"] = "audio_call"
    model_id: Literal["lfm3-audio-2b-bf16"] = "lfm3-audio-2b-bf16"
    mode: Literal["transcribe", "reply"]
    audio: str | None = Field(default=None, max_length=900_000)
    messages: list[ChatMessage] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_turn(self):
        if self.mode == "transcribe":
            if self.audio is None or self.messages:
                raise ValueError(
                    "Audio transcription needs one utterance and no history"
                )
        elif self.audio is not None or not self.messages:
            raise ValueError("Audio replies need text history and no utterance")
        return self


class TranscriptionFormattingTask(TaskInput):
    feature: Literal["stt_formatting"] = "stt_formatting"
    text: str = Field(min_length=1, max_length=32000)


class VideoTask(TaskInput):
    feature: Literal["video"] = "video"
    model: str = "h3-fl2va-int8-turbo"
    seed: int = 42
    prompt: str
    negative_prompt: Literal[""] = ""
    duration: int | None = Field(default=None, ge=1, le=15)
    fps: Literal[24] = 24
    resolution: Literal["480p", "768p"] | None = None
    aspect: Literal["16:9", "9:16", "1:1"] = "16:9"


class DownloadTask(TaskInput):
    feature: Literal["models"] = "models"
    model_id: str
    license_acknowledged: bool = False


class TierSelectionTask(TaskInput):
    feature: Literal["tier_selection"] = "tier_selection"
    tier: TierId


class ModelContextTask(TaskInput):
    feature: Literal["model_context"] = "model_context"
    model_id: str
    context_limit: int | None = Field(ge=1, le=2**31 - 1)


class WhisperSelectionTask(TaskInput):
    feature: Literal["whisper_selection"] = "whisper_selection"
    model: str


class WhisperFormattingTask(TaskInput):
    feature: Literal["whisper_formatting"] = "whisper_formatting"
    enabled: bool


InferenceInput = Annotated[
    ChatTask
    | ImageTask
    | DecisionTask
    | VisionTask
    | SpeechTask
    | TranscriptionTask
    | TranscriptionFormattingTask
    | AudioCallTask
    | VideoTask
    | DownloadTask
    | TierSelectionTask
    | ModelContextTask
    | WhisperSelectionTask
    | WhisperFormattingTask,
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
            "tier_selection",
            "model_context",
            "whisper_selection",
            "whisper_formatting",
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
