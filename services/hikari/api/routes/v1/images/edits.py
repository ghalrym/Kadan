from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Header, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from api.pydantic_models.inference_task import ImageModel
from api.pydantic_models.media import ImageSet
from api.services.inference import inference
from api.services.inference_http import infer

router = APIRouter(prefix="/v1/images/edits", tags=["Images"])


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=8000, pattern=r"\S")
    aspect: Literal["1:1", "4:3", "3:4", "16:9"] = "1:1"
    count: Literal[1, 2, 4] = 4
    seed: int | None = Field(default=None, ge=0, le=2**64 - 1)
    model_id: ImageModel | None = None
    steps: int = Field(default=50, ge=2, le=100)
    image: str = Field(
        min_length=1,
        max_length=4 * ((20 * 1024**2 + 2) // 3) + 22,
        pattern=r"^data:image/png;base64,",
        description="PNG data URL, at most 20 MiB. Dimensions must be multiples of 32, at most 3072 per side and 4,718,592 pixels. Reference inputs must contain at least 65,536 pixels. Source mode uses the source dimensions; reference mode accepts an independent target width and height.",
    )
    mode: Literal["source", "reference"] = Field(
        default="source",
        description="Source edits noise the input image. Reference edits generate a new image conditioned on the reference character and appearance.",
    )
    width: int | None = Field(default=None, ge=32, le=3072, multiple_of=32)
    height: int | None = Field(default=None, ge=32, le=3072, multiple_of=32)
    mask: str | None = Field(
        default=None,
        max_length=4 * ((20 * 1024**2 + 2) // 3) + 22,
        pattern=r"^data:image/png;base64,",
        description="Opaque PNG matching the source size. Black preserves pixels, white edits them; gray blends.",
    )
    strength: float = Field(
        default=0.65,
        ge=0,
        le=1,
        description="Image-to-image noise strength. Zero preserves the source; one regenerates the editable area from noise. Describe the desired result in prompt.",
    )

    @model_validator(mode="after")
    def validate_edit_mode(self):
        if (self.width is None) != (self.height is None):
            raise ValueError(
                "Reference output width and height must be supplied together"
            )
        if self.mode == "reference":
            if self.mask is not None or "strength" in self.model_fields_set:
                raise ValueError("Reference editing does not accept a mask or strength")
        elif self.width is not None:
            raise ValueError("Output dimensions can only be set for reference editing")
        if self.width is not None and self.width * self.height > 4_718_592:
            raise ValueError("Reference output may contain at most 4,718,592 pixels")
        return self


class ImageResponse(BaseModel):
    image: ImageSet
    next_batch_capacity: int | None = Field(
        default=None,
        ge=1,
        le=32,
        description="Advisory image capacity for a subsequent compatible batch; admission is rechecked when it runs.",
    )


@router.post(
    "",
    responses={503: {"description": "Image provider unavailable"}},
    operation_id="editImages",
)
async def create_image(
    body: ImageRequest,
    request: Request,
    x_kadan_request_id: Annotated[UUID | None, Header()] = None,
) -> ImageResponse:
    """Submit source or reference image editing to the shared worker."""
    return await infer(
        request,
        inference.submit(
            body,
            feature="image",
            operation="generate",
            job_id=x_kadan_request_id.hex if x_kadan_request_id is not None else None,
        ),
    )
