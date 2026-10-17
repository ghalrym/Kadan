from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Request

from ....pydantic_models.resources import AgentKey

router = APIRouter()


@router.get("/api/agents/{key}/assets")
async def list_assets(
    key: AgentKey, request: Request, q: Annotated[str, Query(max_length=500)] = ""
):
    return await request.app.state.services.json_request(
        "Tsuchi", "GET", f"/api/agents/{key}/assets", params={"q": q, "trash": "false"}
    )


@router.get("/api/agents/{key}/assets/{asset_id}")
async def get_asset(key: AgentKey, asset_id: UUID, request: Request):
    return await request.app.state.services.json_request(
        "Tsuchi", "GET", f"/api/agents/{key}/assets/{asset_id}"
    )
