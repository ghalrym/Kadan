import base64
import binascii
import json
from io import BytesIO
from typing import Literal

from PIL import Image, UnidentifiedImageError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_BASE64_CHARS = (MAX_IMAGE_BYTES * 4) // 3 + 64
MAX_IMAGE_PIXELS = 16_777_216
MAX_IMAGES_PER_MESSAGE = 16
IMAGE_DATA_URL_PREFIXES = (
    ("data:image/png;base64,", "PNG"),
    ("data:image/jpeg;base64,", "JPEG"),
)


def normalize_image_data_url(url: str) -> str:
    """Validate a PNG or JPEG data URL and normalize it to a PNG data URL."""
    expected_format = None
    payload = None
    for prefix, fmt in IMAGE_DATA_URL_PREFIXES:
        if url.startswith(prefix):
            expected_format = fmt
            payload = url[len(prefix) :]
            break
    if expected_format is None or not payload:
        raise ValueError("Images must be PNG or JPEG data URLs")
    if len(payload) > MAX_IMAGE_BASE64_CHARS:
        raise ValueError("Image exceeds the 20 MiB size limit")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Image data URL is not valid base64") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the 20 MiB size limit")
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format != expected_format:
                raise ValueError(
                    f"Image content does not match declared {expected_format} media type"
                )
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions exceed the 16 megapixel limit")
            image.load()
            if expected_format == "PNG":
                return f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/png;base64,{encoded}"
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Image data is unreadable or invalid") from exc


class ChatFunction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_]+$")
    description: str = ""
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    strict: bool | None = None


class ChatTool(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["function"] = "function"
    function: ChatFunction


class ChatCallFunction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_]+$")
    arguments: dict[str, JsonValue]

    @field_validator(
        "arguments", mode="before", json_schema_input_type=dict[str, JsonValue] | str
    )
    @classmethod
    def parse_arguments(cls, value):
        return json.loads(value) if isinstance(value, str) else value


class ChatToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=120)
    type: Literal["function"] = "function"
    function: ChatCallFunction


class ChatTextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text"]
    text: str


class ChatImageUrl(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str


class ChatImagePart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["image_url"]
    image_url: ChatImageUrl


class ChatMessage(BaseModel):
    """Unified chat message supporting text, images, reasoning, and tool calls."""

    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant", "tool"]
    meta: str | None = None
    text: str = ""
    content: list[ChatTextPart | ChatImagePart] | str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    images: list[str] | None = Field(default=None, max_length=MAX_IMAGES_PER_MESSAGE)
    tool_calls: list[ChatToolCall] | None = None
    tool_call_id: str | None = Field(default=None, min_length=1, max_length=120)
    name: str | None = Field(default=None, min_length=1, max_length=120)
    reasoning_content: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_content(cls, values):
        if not isinstance(values, dict):
            return values
        data = dict(values)
        content = data.get("content")
        if content is not None and data.get("text"):
            raise ValueError("Specify either text or content, not both")
        if isinstance(content, str):
            data["text"] = data.pop("content")
        elif isinstance(content, list):
            images = data.get("images") or []
            if not isinstance(images, list):
                raise ValueError("Images must be a list of image data URLs")  # noqa: TRY004 - Pydantic handles ValueError as input validation.
            count = sum(
                isinstance(part, ChatImagePart)
                or isinstance(part, dict)
                and part.get("type") == "image_url"
                for part in content
            )
            if count + len(images) > MAX_IMAGES_PER_MESSAGE:
                raise ValueError(
                    f"A message may contain at most {MAX_IMAGES_PER_MESSAGE} images"
                )
            if images:
                data["content"] = [
                    *(
                        {"type": "image_url", "image_url": {"url": url}}
                        for url in images
                    ),
                    *content,
                ]
                data.pop("images")
        return data

    @field_validator("images")
    @classmethod
    def normalize_images(cls, value):
        return [normalize_image_data_url(url) for url in value] if value else None

    @field_validator("content")
    @classmethod
    def normalize_parts(cls, value):
        if not isinstance(value, list):
            return value
        return [
            ChatImagePart(
                type="image_url",
                image_url=ChatImageUrl(
                    url=normalize_image_data_url(part.image_url.url)
                ),
            )
            if isinstance(part, ChatImagePart)
            else part
            for part in value
        ]

    def ordered_content(self) -> list[ChatTextPart | ChatImagePart]:
        parts = list(self.content) if isinstance(self.content, list) else []
        if self.text:
            parts.append(ChatTextPart(type="text", text=self.text))
        for image in self.images or ():
            parts.append(
                ChatImagePart(type="image_url", image_url=ChatImageUrl(url=image))
            )
        return parts

    @model_validator(mode="after")
    def valid_role_fields(self):
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError("Only assistant messages may contain tool calls")
        if (self.tool_call_id is not None) != (self.role == "tool"):
            raise ValueError("Tool results require their tool call ID")
        if self.name is not None and self.role != "tool":
            raise ValueError("Only tool messages may specify a tool name")
        parts = self.ordered_content()
        image_count = sum(isinstance(part, ChatImagePart) for part in parts)
        if image_count and self.role != "user":
            raise ValueError("Only user messages may contain images")
        if (
            not any(isinstance(part, ChatTextPart) and part.text for part in parts)
            and not image_count
            and not self.tool_calls
            and self.role != "tool"
        ):
            raise ValueError(
                "A message requires text, an image, or an assistant tool call"
            )
        return self
