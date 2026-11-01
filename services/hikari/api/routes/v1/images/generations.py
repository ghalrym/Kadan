from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Header, Request
from pydantic import BaseModel, ConfigDict, Field

from api.pydantic_models.inference_task import ImageModel
from api.pydantic_models.media import ImageSet
from api.services.inference import inference
from api.services.inference_http import infer
from api.services.subscription_images import image_subscriptions

router = APIRouter(prefix="/v1/images/generations", tags=["Images"])


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["local"] | None = Field(
        default=None,
        exclude=True,
        description="Omit to use the image provider in Settings. Local explicitly preserves native generation, including pet stages.",
    )
    prompt: str = Field(min_length=1, max_length=8000, pattern=r"\S")
    aspect: Literal["1:1", "4:3", "3:4", "16:9"] = "1:1"
    count: Literal[1, 2, 4] = 4
    batch_collection_target: int | None = Field(
        default=None,
        ge=1,
        le=32,
        strict=True,
        description="Expected image count in the submitting cohort. Only shortens collection waiting; queued compatible requests may still share a larger admitted batch.",
    )
    seed: int | None = Field(default=None, ge=0, le=2**64 - 1)
    model_id: ImageModel | None = Field(
        default=None, description="Omit to use the image model selected in Settings."
    )
    steps: int = Field(
        default=50,
        ge=2,
        le=100,
        description="Base model step count. Turbo always uses its saved eight-step schedule.",
    )


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
    operation_id="generateImages",
)
async def create_image(
    body: ImageRequest,
    request: Request,
    x_kadan_request_id: Annotated[UUID | None, Header()] = None,
) -> ImageResponse:
    """Generate through the selected provider, or explicitly use Local."""
    job_id = x_kadan_request_id.hex if x_kadan_request_id else uuid4().hex

    async def local():
        return await inference.submit(
            body,
            feature="image",
            operation="generate",
            job_id=job_id,
        )

    return await infer(request, image_subscriptions.generate(body, job_id, local))
