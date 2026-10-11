from typing import Literal
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from api.pydantic_models.media import ImageSet

router = APIRouter(prefix="/v1/images/edits", tags=["Images"])

class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=8000, pattern=r"\S")
    aspect: Literal["1:1", "4:3", "3:4", "16:9"] = "1:1"
    count: Literal[1, 2, 4] = 4
    seed: int | None = Field(default=None, ge=0)
    image: str = Field(min_length=1, max_length=2048, description="Source image reference; no image provider is configured and this reference is not fetched")
    strength: float = Field(default=0.65, ge=0, le=1)


class ImageResponse(BaseModel):
    image: ImageSet


@router.post("", responses={503: {"description": "Image provider unavailable"}}, operation_id="editImages")
async def create_image(body: ImageRequest, request: Request) -> ImageResponse:
    """Queue validated inference; unavailable providers still return HTTP 503."""
    raise HTTPException(503, 'Native image editing is unavailable.')
