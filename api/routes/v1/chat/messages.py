from fastapi import APIRouter
from pydantic import BaseModel
from api.pydantic_models.chat import ChatMessage

router = APIRouter(prefix="/v1/chat/messages", tags=["Chat"])


class MessagesResponse(BaseModel):
    """Compatibility envelope for client-owned conversations; the server has no history."""
    messages: list[ChatMessage]


@router.get("", operation_id="listMessages", description="Chat is stateless on the server. Conversations are held by the calling client; no persisted messages are available.")
def list_messages() -> MessagesResponse:
    """Return an empty history because conversations live in the calling client.

    This endpoint does not read or persist chat content."""
    return MessagesResponse(messages=[])
