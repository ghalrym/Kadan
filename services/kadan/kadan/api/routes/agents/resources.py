from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Request

from ...pydantic_models.resources import AgentKey, Resource, ResourceKind

router = APIRouter()


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
