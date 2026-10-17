from fastapi import APIRouter, Request

from ..pydantic_models.plugins import PluginManifest

router = APIRouter()


@router.get("/api/plugins")
async def plugins(request: Request) -> list[PluginManifest]:
    return request.app.state.services.plugins.catalog()
