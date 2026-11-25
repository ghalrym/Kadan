from uuid import UUID

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, field_validator

from ....pydantic_models.resources import AgentKey, Title
from ....pydantic_models.space import SpaceItem

router = APIRouter()


class FolderCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: Title
    parent_id: UUID | None = None

    @field_validator("title")
    @classmethod
    def nonblank_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Name the folder")
        return value.strip()


@router.post("/api/agents/{key}/space/folders", status_code=201)
async def create_folder(
    key: AgentKey, value: FolderCreate, request: Request
) -> SpaceItem:
    return await request.app.state.services.space.create_folder(
        key, value.title, value.parent_id
    )
