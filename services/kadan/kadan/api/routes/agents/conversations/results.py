from uuid import UUID, uuid5

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from ....config import ServiceError
from ....pydantic_models.resources import AgentKey, Title, WorkspaceDocument

router = APIRouter()


class SaveResultInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: Title

    @field_validator("title")
    @classmethod
    def nonblank_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Title cannot be blank")
        return value.strip()


@router.post(
    "/api/agents/{key}/conversations/{conversation_id}/messages/{message_id}/document"
)
async def save_result(
    key: AgentKey,
    conversation_id: UUID,
    message_id: UUID,
    value: SaveResultInput,
    request: Request,
) -> WorkspaceDocument:
    document = await request.app.state.services.json_request(
        "Tsuchi",
        "POST",
        f"/api/agents/{key}/resources/{conversation_id}/messages/{message_id}/document",
        json=value.model_dump(mode="json"),
    )
    try:
        saved = WorkspaceDocument.model_validate(document)
    except ValidationError as error:
        raise ServiceError(
            "Tsuchi returned an invalid saved result.", 502, "invalid_response"
        ) from error
    if (
        saved.agent_key != key
        or saved.id != uuid5(conversation_id, f"chat-result:{message_id}")
        or saved.kind != "space"
        or saved.deleted_at is not None
    ):
        raise ServiceError(
            "Tsuchi returned an invalid saved result.", 502, "invalid_response"
        )
    return saved
