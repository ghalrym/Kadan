from uuid import UUID

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ....pydantic_models.resources import AgentKey, Title
from ....pydantic_models.space import SpaceItem, SpacePage

router = APIRouter()
# Tsuchi's limit for a resource's Markdown.
MARKDOWN_LENGTH = 2_000_000


class PageFields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: Title

    @field_validator("title")
    @classmethod
    def nonblank_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Name the page")
        return value.strip()


class PageCreate(PageFields):
    markdown: str = Field(default="", max_length=MARKDOWN_LENGTH)
    parent_id: UUID | None = None


class PageUpdate(PageFields):
    markdown: str = Field(max_length=MARKDOWN_LENGTH)
    expected_version: int = Field(ge=1)


@router.post("/api/agents/{key}/space/pages", status_code=201)
async def create_page(key: AgentKey, value: PageCreate, request: Request) -> SpaceItem:
    return await request.app.state.services.space.create_page(
        key, value.title, value.markdown, value.parent_id
    )


@router.get("/api/agents/{key}/space/pages/{page_id}")
async def get_page(key: AgentKey, page_id: UUID, request: Request) -> SpacePage:
    return await request.app.state.services.space.page(key, page_id)


@router.put("/api/agents/{key}/space/pages/{page_id}")
async def update_page(
    key: AgentKey, page_id: UUID, value: PageUpdate, request: Request
) -> SpacePage:
    return await request.app.state.services.space.update_page(
        key, page_id, value.title, value.markdown, value.expected_version
    )
