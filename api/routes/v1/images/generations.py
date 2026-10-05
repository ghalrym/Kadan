from typing import Literal
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from api.pydantic_models.media import ImageSet
from api.services.images import image_manager
from api.services.runtime import RuntimeFailure

router = APIRouter(prefix="/v1/images/generations", tags=["Images"])

class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = Field(default=None, description="Native checkpoint override; null uses the saved image selection")
    prompt: str = Field(min_length=1, max_length=8000, pattern=r"\S")
    aspect: Literal["1:1", "4:3", "3:4", "16:9"] = "1:1"
    count: Literal[1, 2, 4] = 4
    seed: int | None = Field(default=None, ge=0, le=2**53 - 1)


class ImageResponse(BaseModel):
    image: ImageSet


@router.post("", responses={503: {"description": "Image provider unavailable"}}, operation_id="generateImages")
async def create_image(body: ImageRequest, request: Request) -> ImageResponse:
    """Generate real PNG results with Kadan-owned native inference."""
    try:
        return ImageResponse(image=await image_manager.run(request, body))
    except RuntimeFailure as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
