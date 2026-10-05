from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from api.services.images import image_manager
from api.services.runtime import RuntimeFailure
from pydantic import BaseModel
from api.pydantic_models.media import ImageSet

router = APIRouter(prefix="/v1/images", tags=["Images"])

class ImagesResponse(BaseModel):
    images: list[ImageSet]


@router.get("", operation_id="listImages")
def list_images() -> ImagesResponse:
    """Return metadata for completed native image results."""
    return ImagesResponse(images=image_manager.history())


@router.get('/{identifier}/files/{index}', operation_id='getImageFile')
def image_file(identifier: str, index: int):
    """Serve one published PNG without exposing arbitrary filesystem paths."""
    try:
        return FileResponse(image_manager.file(identifier, index), media_type='image/png')
    except RuntimeFailure as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
