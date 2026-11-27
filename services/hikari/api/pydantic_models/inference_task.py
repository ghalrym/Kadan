"""Serializable jobs submitted to the inference worker's FIFO."""

from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from api.pydantic_models.chat import ChatMessage, ChatTool
from api.pydantic_models.decisions import DecisionQuestion
from api.pydantic_models.inference import Operation

LlmModel = Literal[
    "inkling-2-mini-iq4-nl",
    "inkling-2-mini-q8-0",
    "inkling-2-mini-abliterated-iq4-nl",
    "inkling-2-mini-abliterated-q8-0",
]
WeightVariant = Literal["base", "abliterated"]
# Image inference uses FLUX.2 Klein 4B Distilled.
ImageModel = Literal["flux.2-klein-4b"]


def inkling_model_variant(model_id: str | None) -> WeightVariant | None:
    if model_id in ("inkling-2-mini-iq4-nl", "inkling-2-mini-q8-0"):
        return "base"
    if model_id in (
        "inkling-2-mini-abliterated-iq4-nl",
        "inkling-2-mini-abliterated-q8-0",
    ):
        return "abliterated"
    return None


def inkling_model_quant(model_id: str | None) -> Literal["iq4-nl", "q8-0"] | None:
    if model_id in ("inkling-2-mini-iq4-nl", "inkling-2-mini-abliterated-iq4-nl"):
        return "iq4-nl"
    if model_id in ("inkling-2-mini-q8-0", "inkling-2-mini-abliterated-q8-0"):
        return "q8-0"
    return None


def resolve_inkling_variant_model(
    selected_model_id: str, variant: WeightVariant
) -> LlmModel:
    q8 = inkling_model_quant(selected_model_id) == "q8-0"
    if variant == "abliterated":
        return (
            "inkling-2-mini-abliterated-q8-0"
            if q8
            else "inkling-2-mini-abliterated-iq4-nl"
        )
    return "inkling-2-mini-q8-0" if q8 else "inkling-2-mini-iq4-nl"


def same_inkling_precision_family(model_a: str | None, model_b: str | None) -> bool:
    quant_a = inkling_model_quant(model_a)
    return quant_a is not None and quant_a == inkling_model_quant(model_b)


class TaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatTask(TaskInput):
    feature: Literal["llm"] = "llm"
    messages: list[ChatMessage] = Field(default_factory=list)
    tools: list[ChatTool] | None = Field(default=None, max_length=256)
    model: LlmModel | None = None
    model_id: LlmModel | None = None
    weight_variant: WeightVariant | None = None
    context_limit: int | None = None
    stream: bool = False
    conversation_id: str | None = None
    reuse_prefix: bool = True
    max_tokens: int | None = Field(default=None, ge=1, strict=True)
    reasoning_tokens: int | None = Field(default=None, ge=1, strict=True)
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    priority: Literal["interactive", "background"] = "interactive"

    @model_validator(mode="after")
    def validate_weight_variant(self):
        explicit_model = self.model_id or self.model
        if (
            self.weight_variant is not None
            and explicit_model is not None
            and inkling_model_variant(explicit_model) != self.weight_variant
        ):
            raise ValueError("Requested model does not match requested weight_variant")
        return self


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
    steps: int = 4
    image: str | None = None
    reference_images: list[str] | None = Field(default=None, min_length=1, max_length=3)
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
        else:
            if self.width is not None:
                raise ValueError(
                    "Output dimensions can only be set for reference editing"
                )
            if self.reference_images is not None:
                raise ValueError(
                    "Additional reference images can only be set for reference editing"
                )
        if self.width is not None and self.width * self.height > 4_718_592:
            raise ValueError("Reference output may contain at most 4,718,592 pixels")
        return self


class DecisionTask(TaskInput):
    feature: Literal["decisions"] = "decisions"
    state: str
    questions: list[DecisionQuestion]
    weight_variant: WeightVariant | None = None


class SpeechVoice(TaskInput):
    mode: Literal["custom"]
    speaker: str
    instruction: str = ""


class SpeechTask(TaskInput):
    feature: Literal["tts"] = "tts"
    script: str
    model_id: Literal["qwen-tts-1.7b-custom"] = "qwen-tts-1.7b-custom"
    language: str = "Auto"
    voice: SpeechVoice


class TranscriptionTask(TaskInput):
    feature: Literal["stt"] = "stt"
    audio: str = Field(max_length=2 * 1024**2)
    model: Literal["medium"] | None = None
    language: str | None = None
    formatting: bool | None = None


class AudioCallTask(TaskInput):
    """Bounded buffered turn on the explicitly selected experimental audio worker."""

    feature: Literal["audio_call"] = "audio_call"
    model_id: Literal["lfm3-audio-2b-bf16"] = "lfm3-audio-2b-bf16"
    mode: Literal["transcribe", "reply", "turn"]
    codec_streaming_probe: bool = Field(default=False, strict=True)
    sentence_tts: bool = Field(default=False, strict=True)
    audio: str | None = Field(default=None, max_length=900_000)
    messages: list[ChatMessage] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_turn(self):
        if self.sentence_tts and (self.mode != "turn" or self.codec_streaming_probe):
            raise ValueError(
                "Sentence TTS requires a complete turn without the private codec probe"
            )
        if self.codec_streaming_probe and self.mode != "turn":
            raise ValueError("The private codec probe requires a complete audio turn")
        if self.mode == "transcribe":
            if self.audio is None or self.messages:
                raise ValueError(
                    "Audio transcription needs one utterance and no history"
                )
        elif self.mode == "reply":
            if self.audio is not None or not self.messages:
                raise ValueError("Audio replies need text history and no utterance")
        elif self.audio is None or not self.messages:
            raise ValueError(
                "A complete audio turn needs an utterance and text history"
            )
        return self


class TranscriptionFormattingTask(TaskInput):
    feature: Literal["stt_formatting"] = "stt_formatting"
    text: str = Field(min_length=1, max_length=32000)


class DownloadTask(TaskInput):
    feature: Literal["models"] = "models"
    model_id: str
    license_acknowledged: bool = False


class ModelContextTask(TaskInput):
    feature: Literal["model_context"] = "model_context"
    model_id: str
    context_limit: int | None = Field(ge=1, le=2**31 - 1)


class WhisperFormattingTask(TaskInput):
    feature: Literal["whisper_formatting"] = "whisper_formatting"
    enabled: bool


class LlmSelectionTask(TaskInput):
    feature: Literal["llm_selection"] = "llm_selection"
    model_id: LlmModel


InferenceInput = Annotated[
    ChatTask
    | ImageTask
    | DecisionTask
    | SpeechTask
    | TranscriptionTask
    | TranscriptionFormattingTask
    | AudioCallTask
    | DownloadTask
    | ModelContextTask
    | WhisperFormattingTask
    | LlmSelectionTask,
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
            "model_context",
            "whisper_formatting",
            "llm_selection",
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
