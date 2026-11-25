from typing import Self
from uuid import UUID

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ....conversation_client import DEFAULT_CONVERSATION_TITLE
from ....pydantic_models.conversations import ConversationSummary
from ....pydantic_models.resources import AgentKey, Resource, Title

router = APIRouter()


class ConversationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: Title = DEFAULT_CONVERSATION_TITLE

    @field_validator("title")
    @classmethod
    def nonblank_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Title cannot be blank")
        return value.strip()


class ConversationUpdate(BaseModel):
    """Only fields present in the body change."""

    model_config = ConfigDict(extra="forbid")
    title: Title | None = None

    @field_validator("title")
    @classmethod
    def nonblank_title(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Title cannot be blank")
        return value.strip() if value is not None else None

    @model_validator(mode="after")
    def title_not_null(self) -> Self:
        if "title" in self.model_fields_set and self.title is None:
            raise ValueError("Title cannot be blank")
        return self


@router.get("/api/agents/{key}/conversations")
async def list_conversations(
    key: AgentKey, request: Request
) -> list[ConversationSummary]:
    return await request.app.state.services.conversations.list(key)


@router.post("/api/agents/{key}/conversations", status_code=201)
async def create_conversation(
    key: AgentKey, value: ConversationCreate, request: Request
) -> Resource:
    return await request.app.state.services.conversations.create(key, value.title)


@router.put("/api/agents/{key}/conversations/{conversation_id}")
async def update_conversation(
    key: AgentKey, conversation_id: UUID, value: ConversationUpdate, request: Request
) -> ConversationSummary:
    changes = {name: getattr(value, name) for name in value.model_fields_set}
    return await request.app.state.services.conversations.update(
        key, conversation_id, changes
    )


@router.delete("/api/agents/{key}/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    key: AgentKey, conversation_id: UUID, request: Request
) -> Response:
    await request.app.state.services.conversations.trash(key, conversation_id)
    return Response(status_code=204)
