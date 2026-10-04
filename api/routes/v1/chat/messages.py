from fastapi import APIRouter
from pydantic import BaseModel
from api.pydantic_models.chat import ChatMessage

router = APIRouter(prefix="/v1/chat/messages", tags=["Chat"])


class MessagesResponse(BaseModel):
    messages: list[ChatMessage]


@router.get("", operation_id="listMessages", description="Chat is stateless on the server. Conversations are held by the calling client; no persisted messages are available.")
def list_messages() -> MessagesResponse:
    return MessagesResponse(messages=[])
