from fastapi import APIRouter
from pydantic import BaseModel
from api.pydantic_models.media import ImageSet

router = APIRouter(prefix="/v1/images", tags=["Images"])

MOCK_IMAGE_SETS = [
    ImageSet(
        id='image-set-2',
        mode='Generate',
        prompt='isometric server room at dawn, soft volumetric light, muted palette',
        aspect='landscape',
        seeds=[48213, 48214, 48215, 48216],
        meta='FLUX.1 [dev] · 1152×864 · 14:21',
    ),
    ImageSet(
        id='image-set-1',
        mode='Edit',
        prompt='replace the sky with a stormy overcast, keep the building untouched',
        aspect='wide',
        seeds=[90377, 90378],
        meta='FLUX.1 Kontext · 1344×768 · 14:08',
    ),
]


class ImagesResponse(BaseModel):
    images: list[ImageSet]


@router.get("")
def list_images() -> ImagesResponse:
    return ImagesResponse(images=MOCK_IMAGE_SETS)
