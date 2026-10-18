from uuid import UUID

import anyio
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ....pydantic_models.resources import AgentKey
from .notes import NoteInput

router = APIRouter()


class ChatInput(NoteInput):
    # A null value explicitly chooses no persona; it is never inferred from storage.
    persona_id: UUID | None


class ChatResponse(StreamingResponse):
    async def __call__(self, scope, receive, send):
        # Watch disconnects even while Hikari is silent, such as during a model load.
        # Keep generator cleanup in its streaming task.
        async with anyio.create_task_group() as tasks:

            async def stream():
                try:
                    await self.stream_response(send)
                finally:
                    await self.body_iterator.aclose()
                    tasks.cancel_scope.cancel()

            async def disconnect():
                await self.listen_for_disconnect(receive)
                tasks.cancel_scope.cancel()

            tasks.start_soon(stream)
            tasks.start_soon(disconnect)


@router.post("/api/agents/{key}/conversations/{conversation_id}/chat")
async def chat(
    key: AgentKey, conversation_id: UUID, value: ChatInput, request: Request
):
    service = request.app.state.services
    prepared = await service.prepare_chat(key, conversation_id, value)
    saved = await service.save_chat_user(key, prepared)
    return ChatResponse(
        service.chat_events(key, prepared, saved),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"},
    )
