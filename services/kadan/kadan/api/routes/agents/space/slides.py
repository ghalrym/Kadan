from uuid import UUID

from fastapi import APIRouter, Request

from ....pydantic_models.resources import AgentKey
from ....pydantic_models.space import SpaceSlides

router = APIRouter()


@router.get("/api/agents/{key}/space/slides/{deck_id}")
async def get_slides(key: AgentKey, deck_id: UUID, request: Request) -> SpaceSlides:
    return await request.app.state.services.space.slides(key, deck_id)
