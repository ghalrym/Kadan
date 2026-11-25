from uuid import UUID

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...pydantic_models.conversations import Space
from ...pydantic_models.resources import AgentKey, Title

router = APIRouter()


class SpaceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: Title
    instructions: str = Field(default="", max_length=48000)

    @field_validator("title")
    @classmethod
    def nonblank_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Name the Space")
        return value.strip()


class SpaceUpdate(SpaceCreate):
    instructions: str = Field(max_length=48000)
    expected_version: int = Field(ge=1)


@router.get("/api/agents/{key}/spaces")
async def list_spaces(key: AgentKey, request: Request) -> list[Space]:
    return await request.app.state.services.conversations.spaces(key)


@router.post("/api/agents/{key}/spaces", status_code=201)
async def create_space(key: AgentKey, value: SpaceCreate, request: Request) -> Space:
    return await request.app.state.services.conversations.create_space(
        key, value.title, value.instructions
    )


@router.get("/api/agents/{key}/spaces/{space_id}")
async def get_space(key: AgentKey, space_id: UUID, request: Request) -> Space:
    return await request.app.state.services.conversations.space(key, space_id)


@router.put("/api/agents/{key}/spaces/{space_id}")
async def update_space(
    key: AgentKey, space_id: UUID, value: SpaceUpdate, request: Request
) -> Space:
    return await request.app.state.services.conversations.update_space(
        key, space_id, value.title, value.instructions, value.expected_version
    )


@router.delete("/api/agents/{key}/spaces/{space_id}", status_code=204)
async def delete_space(key: AgentKey, space_id: UUID, request: Request) -> Response:
    await request.app.state.services.conversations.delete_space(key, space_id)
    return Response(status_code=204)
