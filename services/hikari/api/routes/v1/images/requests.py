from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.services.inference import inference
from api.services.subscription_images import image_subscriptions

router = APIRouter(prefix="/v1/images/requests", tags=["Images"])


class CancellationResponse(BaseModel):
    status: Literal["cancelled", "not_active"]


@router.post("/{request_id}/cancel", operation_id="cancelImageRequest")
async def cancel_image_request(request_id: UUID) -> CancellationResponse:
    """Join an active image request's cleanup before acknowledging cancellation.

    An absent request has no recoverable result here. Callers must distinguish
    that from a confirmed cancellation and reconcile their durable output first.
    """
    if await image_subscriptions.cancel(request_id.hex):
        return CancellationResponse(status="cancelled")
    job = inference.worker.jobs.get(request_id.hex)
    if job is None:
        return CancellationResponse(status="not_active")
    if job.request.feature != "image":
        raise HTTPException(409, "This request is not an image request.")
    await inference.cancel(request_id.hex)
    return CancellationResponse(status="cancelled")
