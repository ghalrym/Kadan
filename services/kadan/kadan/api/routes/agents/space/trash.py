from fastapi import APIRouter, Request

from ....pydantic_models.resources import AgentKey
from ....pydantic_models.space import SpaceItem

router = APIRouter()


@router.get("/api/agents/{key}/space/trash")
async def list_trash(key: AgentKey, request: Request) -> list[SpaceItem]:
    return await request.app.state.services.space.trash(key)
