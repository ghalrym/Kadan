from uuid import UUID

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ....pydantic_models.resources import AgentKey
from .notes import NoteInput

router = APIRouter()


class ChatInput(NoteInput):
    # A null value explicitly chooses no persona; it is never inferred from storage.
    persona_id: UUID | None


@router.post("/api/agents/{key}/conversations/{conversation_id}/chat")
async def chat(
    key: AgentKey, conversation_id: UUID, value: ChatInput, request: Request
):
    service = request.app.state.services
    prepared = await service.prepare_chat(key, conversation_id, value)
    saved = await service.save_chat_user(key, prepared)
    return StreamingResponse(
        service.chat_events(key, prepared, saved),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"},
    )
