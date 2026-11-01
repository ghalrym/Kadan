"""Screenshot understanding through the selected local or subscription provider."""

import asyncio
from typing import Literal

from experimental_worker.vision import image_bytes
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.repositories import subscriptions
from api.services.inference import inference
from api.services.inference_errors import InferenceFailure
from api.services.inference_http import infer
from api.services.subscription_vision import SubscriptionVision

router = APIRouter(prefix="/v1/vision", tags=["Vision"])


class DescribeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["lfm3-vl-2b-bf16"] = "lfm3-vl-2b-bf16"
    image_base64: str = Field(min_length=1, max_length=2_796_204)
    media_type: Literal["image/png", "image/jpeg"]
    question: str = Field(min_length=1)
    max_new_tokens: int | None = Field(default=None, ge=1, lt=128000, strict=True)

    @field_validator("question")
    @classmethod
    def question_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A screenshot question cannot be blank.")
        return value


class DescribeResponse(BaseModel):
    model: Literal["lfm3-vl-2b-bf16", "claude-haiku-5-5", "gpt-6.1-terra"]
    text: str = Field(min_length=1)


@router.post("/describe", operation_id="describeScreenshot")
async def describe_screenshot(
    body: DescribeRequest, request: Request
) -> DescribeResponse:
    try:
        image_bytes(body.model_dump())
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    try:
        subscription = await asyncio.to_thread(subscriptions.active, "vision")
    except subscriptions.SubscriptionError as error:
        raise HTTPException(409, str(error)) from error
    if subscription is not None:
        try:
            stream = SubscriptionVision(subscription, body)
        except InferenceFailure as error:
            raise HTTPException(error.status_code, str(error)) from error
        text = await infer(request, stream.text())
        if not text.strip():
            raise HTTPException(502, "Vision provider returned an empty description.")
        return DescribeResponse(model=stream.model, text=text)
    # Queue serialization omits unset defaults; the dedicated vision endpoint
    # still needs to carry its explicit checkpoint to the experimental worker.
    selected = body.model_copy(update={"model": body.model})
    result = await infer(request, inference.submit(selected, feature="vision"))
    try:
        response = DescribeResponse.model_validate(result)
        if response.model != body.model:
            raise ValueError("Unexpected local vision model")
        return response
    except ValueError as error:
        raise HTTPException(
            502, "Vision worker returned an invalid description."
        ) from error
