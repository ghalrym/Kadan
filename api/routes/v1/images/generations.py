from typing import Literal
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field
from api.pydantic_models.media import ImageSet

router = APIRouter(prefix="/v1/images/generations", tags=["Images"])

MOCK_IMAGE_SET = ImageSet(
    id='image-set-2',
    mode='Generate',
    prompt='isometric server room at dawn, soft volumetric light, muted palette',
    aspect='landscape',
    seeds=[48213, 48214, 48215, 48216],
    meta='FLUX.1 [dev] · 1152×864 · 14:21',
)


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)
    aspect: Literal["1:1", "4:3", "3:4", "16:9"] = "1:1"
    count: Literal[1, 2, 4] = 4
    seed: int | None = Field(default=None, ge=0)


class ImageResponse(BaseModel):
    image: ImageSet


@router.post("")
def create_image(body: ImageRequest) -> ImageResponse:
    return ImageResponse(image=MOCK_IMAGE_SET)
