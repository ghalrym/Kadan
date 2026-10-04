from typing import Literal
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field
from api.pydantic_models.media import ImageSet

router = APIRouter(prefix="/v1/images/edits", tags=["Images"])

MOCK_IMAGE_SET = ImageSet(
    id='image-set-1',
    mode='Edit',
    prompt='replace the sky with a stormy overcast, keep the building untouched',
    aspect='wide',
    seeds=[90377, 90378],
    meta='FLUX.1 Kontext · 1344×768 · 14:08',
)


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)
    aspect: Literal["1:1", "4:3", "3:4", "16:9"] = "1:1"
    count: Literal[1, 2, 4] = 4
    seed: int | None = Field(default=None, ge=0)
    image: str = Field(min_length=1, description="Source image reference; not fetched in mock mode")
    strength: float = Field(default=0.65, ge=0, le=1)


class ImageResponse(BaseModel):
    image: ImageSet


@router.post("")
def create_image(body: ImageRequest) -> ImageResponse:
    return ImageResponse(image=MOCK_IMAGE_SET)
