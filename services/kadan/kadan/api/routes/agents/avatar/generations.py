from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from ....pydantic_models.assets import Asset
from ....pydantic_models.resources import AgentKey

router = APIRouter()


class AvatarGenerationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=8000, pattern=r"\S")


@router.post(
    "/api/agents/{key}/avatar/generations", response_model=Asset, status_code=201
)
async def generate_avatar(
    key: AgentKey, value: AvatarGenerationInput, request: Request
):
    return await request.app.state.services.generate_avatar(key, value.prompt)
