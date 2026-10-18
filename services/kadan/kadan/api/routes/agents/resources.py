from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...pydantic_models.resources import AgentKey, Resource, ResourceKind, Title

router = APIRouter()


class ResourceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Each agent has one conversation, created by the chain route.
    kind: Literal["persona"]
    title: Title
    markdown: str = Field(default="", max_length=48000)
    parent_id: UUID | None = None
    tags: list[Annotated[str, Field(min_length=1, max_length=100)]] = Field(
        default_factory=list, max_length=100
    )

    @field_validator("title")
    @classmethod
    def nonblank_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Title cannot be blank")
        return value.strip()


class PersonaUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: Title
    markdown: str = Field(max_length=48000)
    expected_version: int = Field(ge=1)

    @field_validator("title")
    @classmethod
    def nonblank_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Title cannot be blank")
        return value.strip()


@router.get("/api/agents/{key}/resources")
async def list_resources(
    key: AgentKey,
    request: Request,
    kind: ResourceKind | None = None,
    q: Annotated[str, Query(max_length=500)] = "",
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    params = {"q": q, "limit": limit, "offset": offset, "trash": "false"}
    if kind is not None:
        params["kind"] = kind.value
    return await request.app.state.services.json_request(
        "Tsuchi", "GET", f"/api/agents/{key}/resources", params=params
    )


@router.get("/api/agents/{key}/resources/{resource_id}")
async def get_resource(key: AgentKey, resource_id: UUID, request: Request) -> Resource:
    return await request.app.state.services.resource(key, resource_id)


@router.post("/api/agents/{key}/resources", status_code=201)
async def create_resource(key: AgentKey, value: ResourceCreate, request: Request):
    return await request.app.state.services.json_request(
        "Tsuchi",
        "POST",
        f"/api/agents/{key}/resources",
        json=value.model_dump(mode="json"),
    )


@router.put("/api/agents/{key}/resources/{resource_id}")
async def update_persona(
    key: AgentKey, resource_id: UUID, value: PersonaUpdate, request: Request
) -> Resource:
    return await request.app.state.services.update_persona(key, resource_id, value)
