import asyncio
from typing import Self
from uuid import UUID

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, model_validator

from ....pydantic_models.agent_runs import (
    AgentRun,
    ImageJob,
    RequestReference,
    RunContext,
)
from ....pydantic_models.resources import AgentKey, Resource
from .notes import NoteInput

router = APIRouter()


class RunInput(NoteInput):
    references: list[RequestReference] = Field(default_factory=list, max_length=3)
    persona_id: UUID | None
    workflow_id: UUID | None = None
    workflow_version: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def workflow_selection(self) -> Self:
        if (self.workflow_id is None) != (self.workflow_version is None):
            raise ValueError("Select a workflow and its version together")
        return self


class RunCreated(BaseModel):
    run: AgentRun
    conversation: Resource


class ConversationRuns(BaseModel):
    runs: list[AgentRun]
    images: list[ImageJob]
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
        services.resource(key, conversation_id),
    )
    return ConversationRuns(
        runs=sorted(runs, key=lambda run: (run["created_at"], run["id"])),
        images=images,
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


@router.get("/api/agents/{key}/conversations/{conversation_id}/runs/{run_id}/context")
async def run_context(
    key: AgentKey, conversation_id: UUID, run_id: UUID, request: Request
) -> RunContext:
    return await request.app.state.services.runs.selection_context(
        key, conversation_id, run_id
    )
