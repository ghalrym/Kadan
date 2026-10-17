from uuid import UUID

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict

from ....pydantic_models.agents import Agent
from ....pydantic_models.resources import AgentKey

router = APIRouter()


class AvatarInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset_id: UUID | None


@router.put("/api/agents/{key}/avatar", response_model=Agent)
async def select_avatar(key: AgentKey, value: AvatarInput, request: Request):
    return await request.app.state.services.select_avatar(key, value.asset_id)
