from uuid import UUID

from fastapi import APIRouter, Query, Request

from ...coding_client import CodingClient
from ...pydantic_models.coding import CodingJob, CodingStatus

router = APIRouter(prefix="/api/computer/coding")


@router.get("")
async def coding_status(request: Request) -> CodingStatus:
    return await CodingClient(request.app.state.services.sandbox).status()


@router.get("/{job_id}")
async def coding_job(
    request: Request, job_id: UUID, after: int = Query(0, ge=0)
) -> CodingJob:
    return await CodingClient(request.app.state.services.sandbox).job(job_id, after)


@router.post("/{job_id}/cancel")
async def cancel_coding(request: Request, job_id: UUID) -> CodingJob:
    return await CodingClient(request.app.state.services.sandbox).cancel(job_id)
