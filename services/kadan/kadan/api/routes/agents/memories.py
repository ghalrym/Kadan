from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...pydantic_models.resources import AgentKey, Memory, ResourceKind, Title

router = APIRouter()
MEMORY_KINDS = (ResourceKind.MEMORY,)


class MemoryUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: Title
    markdown: str = Field(max_length=2000000)
    expected_version: int = Field(ge=1)

    @field_validator("title")
    @classmethod
    def nonblank_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Title cannot be blank")
        return value.strip()


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


@router.put("/api/agents/{key}/memories/{resource_id}")
async def update_memory(
    key: AgentKey, resource_id: UUID, value: MemoryUpdate, request: Request
) -> Memory:
    return await request.app.state.services.update_memory(key, resource_id, value)
