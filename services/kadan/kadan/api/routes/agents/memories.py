from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Request

from ...pydantic_models.resources import AgentKey, Memory, ResourceKind

router = APIRouter()
MEMORY_KINDS = (ResourceKind.MEMORY,)


@router.get("/api/agents/{key}/memories")
async def list_memories(
    key: AgentKey,
    request: Request,
    q: Annotated[str, Query(max_length=500)] = "",
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[Memory]:
    resources = await request.app.state.services.list_resource_kinds(
        key, MEMORY_KINDS, q=q, limit=limit, offset=offset
    )
    return [Memory.model_validate(resource.model_dump()) for resource in resources]


@router.get("/api/agents/{key}/memories/{resource_id}")
async def get_memory(key: AgentKey, resource_id: UUID, request: Request) -> Memory:
    resource = await request.app.state.services.resource_of_kinds(
        key, resource_id, MEMORY_KINDS
    )
    return Memory.model_validate(resource.model_dump())
