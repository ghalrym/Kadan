from fastapi import APIRouter
from pydantic import BaseModel
from api.pydantic_models.media import ImageSet

router = APIRouter(prefix="/v1/images", tags=["Images"])

class ImagesResponse(BaseModel):
    images: list[ImageSet]


@router.get("", operation_id="listImages")
def list_images() -> ImagesResponse:
    return ImagesResponse(images=[])
