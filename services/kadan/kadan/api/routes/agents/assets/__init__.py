from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, Form, Query, Request, UploadFile

from ....asset_uploads import upload_asset as store_asset
from ....pydantic_models.assets import Asset
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


@router.post("/api/agents/{key}/assets", status_code=201)
async def upload_asset(
    key: AgentKey,
    request: Request,
    asset_id: Annotated[UUID, Form()],
    file: Annotated[UploadFile, File()],
    title: Annotated[
        str | None, Form(min_length=1, max_length=500, pattern=r"\S")
    ] = None,
) -> Asset:
    try:
        return await store_asset(request.app.state.services, key, asset_id, file, title)
    finally:
        await file.close()
