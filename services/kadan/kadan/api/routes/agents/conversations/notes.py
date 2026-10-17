from uuid import UUID, uuid4

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ....pydantic_models.resources import AgentKey, Resource

router = APIRouter()


class NoteInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(min_length=1, max_length=32000)
    expected_version: int = Field(ge=1)
    message_id: UUID = Field(default_factory=uuid4)

    @field_validator("content")
    @classmethod
    def nonblank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be blank")
        return value


@router.post("/api/agents/{key}/conversations/{conversation_id}/notes")
async def save_note(
    key: AgentKey, conversation_id: UUID, value: NoteInput, request: Request
) -> Resource:
    return await request.app.state.services.save_note(key, conversation_id, value)
