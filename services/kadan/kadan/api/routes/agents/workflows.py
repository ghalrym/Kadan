"""Instruction templates are ordinary versioned Space documents."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ...config import ServiceError
from ...pydantic_models.resources import AgentKey, Resource, ResourceKind

router = APIRouter()


class WorkflowBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: int = Field(ge=1, le=1)
    title: str = Field(min_length=1, max_length=500, pattern=r"\S")
    instructions: str = Field(min_length=1, max_length=12000, pattern=r"\S")


class WorkflowPage(BaseModel):
    items: list[Resource]
    next_offset: int | None


def check_workflow(resource: Resource) -> Resource:
    if (
        resource.kind != ResourceKind.SPACE
        or "workflow" not in resource.tags
        or resource.deleted_at is not None
    ):
        raise ServiceError("Workflow not found.", 404)
    return resource


@router.get("/api/agents/{key}/workflows")
async def list_workflows(
    key: AgentKey,
    request: Request,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> WorkflowPage:
    rows = await request.app.state.services.list_resource_kinds(
        key, (ResourceKind.SPACE,), limit=200, offset=offset
    )
    return WorkflowPage(
        items=[row for row in rows if "workflow" in row.tags],
        next_offset=offset + 200 if len(rows) == 200 else None,
    )


@router.get("/api/agents/{key}/workflows/{workflow_id}")
async def get_workflow(key: AgentKey, workflow_id: UUID, request: Request) -> Resource:
    resource = await request.app.state.services.resource_of_kinds(
        key, workflow_id, (ResourceKind.SPACE,)
    )
    return check_workflow(resource)


@router.post("/api/agents/{key}/workflows", status_code=201)
async def create_workflow(
    key: AgentKey, value: WorkflowBundle, request: Request
) -> Resource:
    services = request.app.state.services
    result = await services.json_request(
        "Tsuchi",
        "POST",
        f"/api/agents/{key}/resources",
        json={
            "kind": "space",
            "title": value.title,
            "markdown": value.instructions,
            "tags": ["workflow"],
        },
    )
    # Return the confirmed durable record, never an optimistic local template.
    try:
        resource = check_workflow(Resource.model_validate(result))
    except (ValidationError, ServiceError) as error:
        raise ServiceError(
            "Workflow creation could not be confirmed. Check the list before saving again.",
            502,
        ) from error
    if (
        resource.agent_key != key
        or resource.title != value.title.strip()
        or resource.markdown != value.instructions
        or resource.version != 1
    ):
        raise ServiceError(
            "Workflow creation could not be confirmed. Check the list before saving again.",
            502,
        )
    return resource
