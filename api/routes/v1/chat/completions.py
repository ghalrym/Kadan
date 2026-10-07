from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from api.pydantic_models.chat import ChatMessage
from api.memory_manager import memory_manager
from api.memory_manager.http import infer

router = APIRouter(prefix='/v1/chat/completions', tags=['Chat'])


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    messages: list[ChatMessage] = Field(min_length=1)
    model: str | None = None


class CompletionResponse(BaseModel):
    message: ChatMessage


@router.post('', operation_id='createCompletion')
async def create_completion(body: CompletionRequest, request: Request) -> CompletionResponse:
    """Queue a reply; disconnect waits for native cancellation and cleanup."""
    text = await infer(request, memory_manager.submit(body, feature='llm'))
    return CompletionResponse(message=ChatMessage(role='assistant', text=text))
