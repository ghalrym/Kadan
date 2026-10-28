"""Opt-in screenshot understanding, routed through the existing worker queue."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.services.inference import inference
from api.services.inference_http import infer
from experimental_worker.vision import image_bytes

router = APIRouter(prefix="/v1/vision", tags=["Vision"])


class DescribeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["lfm3-vl-2b-bf16"] = "lfm3-vl-2b-bf16"
    image_base64: str = Field(min_length=1, max_length=2_796_204)
    media_type: Literal["image/png", "image/jpeg"]
    question: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=1024, ge=1, le=1024, strict=True)

    @field_validator("question")
    @classmethod
    def question_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A screenshot question cannot be blank.")
        return value


class DescribeResponse(BaseModel):
    model: Literal["lfm3-vl-2b-bf16"]
    text: str = Field(min_length=1, max_length=32000)


@router.post("/describe", operation_id="describeScreenshot")
async def describe_screenshot(
    body: DescribeRequest, request: Request
) -> DescribeResponse:
    try:
        image_bytes(body.model_dump())
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    # Queue serialization omits unset defaults; the dedicated vision endpoint
    # still needs to carry its explicit checkpoint to the experimental worker.
    selected = body.model_copy(update={"model": body.model})
    result = await infer(request, inference.submit(selected, feature="vision"))
    try:
        return DescribeResponse.model_validate(result)
    except ValueError as error:
        raise HTTPException(
            502, "Vision worker returned an invalid description."
        ) from error
