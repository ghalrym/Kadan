import base64
import binascii
from io import BytesIO
from typing import Literal

from PIL import Image, UnidentifiedImageError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
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
    if not isinstance(url, str):
        raise ValueError("Image URL must be a string")
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
        if "text" in values and "content" in values:
            raise ValueError("Specify either text or content, not both")
        data = dict(values)
        extracted_images: list[str] = []
        if "content" in data:
            content = data.pop("content")
            if isinstance(content, str):
                data["text"] = content
            elif isinstance(content, list):
                text_parts: list[str] = []
                for item in content:
                    if not isinstance(item, dict):
                        raise ValueError("Content parts must be JSON objects")
                    kind = item.get("type")
                    if kind == "text":
                        part = ChatTextPart.model_validate(item)
                        text_parts.append(part.text)
                    elif kind == "image_url":
                        part = ChatImagePart.model_validate(item)
                        extracted_images.append(part.image_url.url)
                    else:
                        raise ValueError(f"Unsupported content part type: {kind!r}")
                data["text"] = "".join(text_parts)
            else:
                raise ValueError("Message content must be a string or list of parts")
        if "images" in data or extracted_images:
            raw_images = data.get("images")
            if raw_images is None:
                combined = extracted_images
            elif isinstance(raw_images, list):
                combined = [*raw_images, *extracted_images]
            else:
                raise ValueError("images must be a list of image data URLs")
            if len(combined) > MAX_IMAGES_PER_MESSAGE:
                raise ValueError(
                    f"A message may contain at most {MAX_IMAGES_PER_MESSAGE} images"
                )
            data["images"] = (
                [normalize_image_data_url(url) for url in combined]
                if combined
                else None
            )
        return data

    @model_validator(mode="after")
    def valid_role_fields(self):
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError("Only assistant messages may contain tool calls")
        if (self.tool_call_id is not None) != (self.role == "tool"):
            raise ValueError("Tool results require their tool call ID")
        if self.name is not None and self.role != "tool":
            raise ValueError("Only tool messages may specify a tool name")
        if self.images and self.role != "user":
            raise ValueError("Only user messages may contain images")
        if (
            not self.text
            and not self.images
            and not self.tool_calls
            and self.role != "tool"
        ):
            raise ValueError(
                "A message requires text, an image, or an assistant tool call"
            )
        return self
