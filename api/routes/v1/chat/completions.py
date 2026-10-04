import asyncio
from contextlib import suppress
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from api.pydantic_models.chat import ChatMessage
from api.services.runtime import RuntimeFailure, runtime_manager

router = APIRouter(prefix='/v1/chat/completions', tags=['Chat'])


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    messages: list[ChatMessage] = Field(min_length=1, max_length=128)
    model: str | None = None


class CompletionResponse(BaseModel):
    message: ChatMessage


@router.post('', operation_id='createCompletion')
async def create_completion(body: CompletionRequest, request: Request) -> CompletionResponse:
    if sum(len(message.text) for message in body.messages) > 32768:
        raise HTTPException(413, 'Conversation exceeds the 32768-character request limit.')
    async def watch_disconnect():
        # FastAPI has already consumed/validated the JSON body. Wait directly on
        # the ASGI channel: is_disconnected() uses an AnyIO cancellation scope
        # that can swallow this task's cancellation during response cleanup.
        while True:
            if (await request.receive())['type'] == 'http.disconnect':
                return

    generation = asyncio.create_task(runtime_manager.complete(body.messages, body.model))
    disconnected = asyncio.create_task(watch_disconnect())
    try:
        done, _ = await asyncio.wait([generation, disconnected], return_when=asyncio.FIRST_COMPLETED)
        if generation not in done:
            generation.cancel()
            raise HTTPException(499, 'Client disconnected; generation cancelled.')
        text = await generation
        return CompletionResponse(message=ChatMessage(role='assistant', text=text))
    except RuntimeFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    finally:
        for task in (generation, disconnected):
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
