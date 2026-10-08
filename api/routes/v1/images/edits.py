from api.memory_manager import memory_manager
from api.memory_manager.http import infer
from typing import Literal
from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field
from api.pydantic_models.media import ImageSet

router = APIRouter(prefix="/v1/images/edits", tags=["Images"])

class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=8000, pattern=r"\S")
    aspect: Literal["1:1", "4:3", "3:4", "16:9"] = "1:1"
    count: Literal[1, 2, 4] = 4
    seed: int | None = Field(default=None, ge=0, le=2**53 - 1)
    image: str = Field(min_length=1, max_length=28_000_000, description="Inline PNG, JPEG or WebP data URL; remote URLs are not fetched. The complete queued JSON request must also fit 16 MiB, so the effective encoded image limit is smaller than 16 MiB after prompt/JSON overhead (about 12 MiB decoded).")



class ImageResponse(BaseModel):
    image: ImageSet


@router.post("", responses={503: {"description": "Image provider unavailable"}}, operation_id="editImages")
async def create_image(body: ImageRequest, request: Request) -> ImageResponse:
    """Queue native Qwen-Image inference under shared memory ownership."""
    return await infer(request, memory_manager.submit(body, feature='image', operation='edit'))
