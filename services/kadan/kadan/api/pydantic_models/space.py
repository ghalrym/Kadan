from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from .resources import Slide

SpaceItemType = Literal["folder", "page", "slides", "image", "file"]


class SpaceItem(BaseModel):
    """One thing in an agent's Space library: a Tsuchi folder, page, deck or file."""

    id: UUID
    type: SpaceItemType
    title: str
    # A resource's parent, or the folder holding a file; null at the top level.
    parent_id: UUID | None
    pinned: bool
    filename: str | None
    media_type: str | None
    size_bytes: int | None
    version: int
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None


class SpacePage(SpaceItem):
    markdown: str


class SpaceSlides(SpaceItem):
    slides: list[Slide]
