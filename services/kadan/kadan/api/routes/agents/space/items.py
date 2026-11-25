from typing import Self
from uuid import UUID

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ....pydantic_models.resources import AgentKey, Title
from ....pydantic_models.space import SpaceItem

router = APIRouter()


class ItemUpdate(BaseModel):
    """Only fields present in the body change; a null parent_id moves to the top."""

    model_config = ConfigDict(extra="forbid")
    title: Title | None = None
    parent_id: UUID | None = None
    pinned: bool | None = None

    @field_validator("title")
    @classmethod
    def nonblank_title(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Title cannot be blank")
        return value.strip() if value is not None else None

    @model_validator(mode="after")
    def no_null_title_or_pin(self) -> Self:
        if "title" in self.model_fields_set and self.title is None:
            raise ValueError("Title cannot be blank")
        if "pinned" in self.model_fields_set and self.pinned is None:
            raise ValueError("Choose whether the item is pinned")
        return self


@router.get("/api/agents/{key}/space/items")
async def list_items(key: AgentKey, request: Request) -> list[SpaceItem]:
    return await request.app.state.services.space.items(key)


@router.put("/api/agents/{key}/space/items/{item_id}")
async def update_item(
    key: AgentKey, item_id: UUID, value: ItemUpdate, request: Request
) -> SpaceItem:
    changes = {name: getattr(value, name) for name in value.model_fields_set}
    return await request.app.state.services.space.update(key, item_id, changes)


@router.delete("/api/agents/{key}/space/items/{item_id}", status_code=204)
async def delete_item(key: AgentKey, item_id: UUID, request: Request) -> Response:
    await request.app.state.services.space.delete(key, item_id)
    return Response(status_code=204)


@router.post("/api/agents/{key}/space/items/{item_id}/restore")
async def restore_item(key: AgentKey, item_id: UUID, request: Request) -> SpaceItem:
    return await request.app.state.services.space.restore(key, item_id)
