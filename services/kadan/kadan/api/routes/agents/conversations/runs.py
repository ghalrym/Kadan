import asyncio
from uuid import UUID

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ....config import ServiceError
from ....pydantic_models.agent_runs import AgentRun, ImageJob
from ....pydantic_models.conversation_work import AgentRunSummary, ImageJobSummary
from ....pydantic_models.resources import AgentKey, Resource, ResourceKind
from .notes import NoteInput

router = APIRouter()


class RunInput(NoteInput):
    persona_id: UUID | None


class RunCreated(BaseModel):
    run: AgentRun
    conversation: Resource


class ConversationRuns(BaseModel):
    runs: list[AgentRunSummary]
    images: list[ImageJobSummary]
    conversation: Resource


@router.post("/api/agents/{key}/conversations/{conversation_id}/runs", status_code=202)
async def create_run(
    key: AgentKey, conversation_id: UUID, value: RunInput, request: Request
) -> RunCreated:
    services = request.app.state.services
    run = await services.runs.enqueue(key, conversation_id, value)
    return RunCreated(
        run=run, conversation=await services.resource(key, conversation_id)
    )


@router.get("/api/agents/{key}/conversations/{conversation_id}/runs")
async def conversation_runs(
    key: AgentKey, conversation_id: UUID, request: Request
) -> ConversationRuns:
    services = request.app.state.services
    runs, images, conversation = await asyncio.gather(
        services.json_request(
            "Tsuchi", "GET", f"/api/agents/{key}/resources/{conversation_id}/runs"
        ),
        services.json_request(
            "Tsuchi", "GET", f"/api/agents/{key}/resources/{conversation_id}/image-jobs"
        ),
        services.resource_of_kinds(key, conversation_id, (ResourceKind.CONVERSATION,)),
    )
    if not isinstance(runs, list) or not isinstance(images, list):
        raise ServiceError(
            "Tsuchi returned an invalid work collection.", 502, "invalid_response"
        )
    run_records = [AgentRun.model_validate(value) for value in runs]
    image_records = [ImageJob.model_validate(value) for value in images]
    if any(
        record.agent_key != key or record.conversation_id != conversation_id
        for record in [*run_records, *image_records]
    ):
        raise ServiceError(
            "Tsuchi returned work outside this conversation.", 502, "invalid_response"
        )
    return ConversationRuns(
        runs=[
            AgentRunSummary.model_validate(run)
            for run in sorted(run_records, key=lambda run: (run.created_at, run.id))
        ],
        images=[ImageJobSummary.model_validate(image) for image in image_records],
        conversation=conversation,
    )


@router.post("/api/agents/{key}/runs/{run_id}/cancel")
async def cancel_run(key: AgentKey, run_id: UUID, request: Request) -> AgentRun:
    return await request.app.state.services.runs.cancel(key, run_id)


@router.post("/api/agents/{key}/image-jobs/{job_id}/cancel")
async def cancel_image(key: AgentKey, job_id: UUID, request: Request) -> ImageJob:
    services = request.app.state.services
    value = await services.json_request(
        "Tsuchi", "POST", f"/api/agents/{key}/image-jobs/{job_id}/cancel", json={}
    )
    return ImageJob.model_validate(value)
