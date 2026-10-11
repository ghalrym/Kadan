import hashlib
import json
import time
from typing import Literal
from uuid import uuid4

import anyio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from api.pydantic_models.chat import ChatMessage
from api.memory_manager import memory_manager
from api.memory_manager.http import infer
from api.inference.errors import InferenceFailure

router = APIRouter(prefix='/v1/chat/completions', tags=['Chat'])


class CompletionMessage(ChatMessage):
    text: str = Field(min_length=1, validation_alias=AliasChoices('text', 'content'))


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    messages: list[CompletionMessage] = Field(min_length=1)
    model: str | None = None
    stream: bool = False
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128,
        description="Client-owned opaque conversation ID; omitted IDs never retain prefix state.")
    reuse_prefix: bool = Field(default=True, description="Disable to compare against full prefill.")


class AssistantMessage(BaseModel):
    role: Literal['assistant'] = 'assistant'
    content: str


class CompletionChoice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: Literal['stop', 'length'] = 'stop'


class CacheUsage(BaseModel):
    hit: bool
    reused_tokens: int
    stored_tokens: int
    host_bytes: int
    device_bytes: dict[str, int]
    reason: str
    retention_reason: str = "unknown"
    limit_bytes: int = 0


class CompletionResponse(BaseModel):
    # Preserve the original Kadan response while adding the standard envelope.
    message: ChatMessage
    id: str
    object: Literal['chat.completion'] = 'chat.completion'
    created: int
    model: str
    choices: list[CompletionChoice]
    cache: CacheUsage | None = None


class OwnedStreamResponse(StreamingResponse):
    def __init__(self, content, owner):
        super().__init__(content, media_type='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
        self.owner = owner

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette uses cancellation scopes on disconnect. Shield cleanup
            # through the queue/native owner, including a disconnect before iteration.
            with anyio.CancelScope(shield=True):
                await self.owner.aclose()


def event_frame(value):
    return 'data: ' + json.dumps(value, ensure_ascii=False) + '\n\n'


def observe_generation(measurement, event):
    """Copy numeric native measurements and safe cache metadata; never output text."""
    timing = event.get('timing') or {}
    for key in ('generation_ttft_ms', 'prefill_ms', 'decode_tokens_per_second', 'output_tokens', 'prefill_tokens'):
        if key in timing:
            measurement[key] = timing[key]
    cache = event.get('cache') or {}
    for source, target in (('hit', 'cache_hit'), ('reused_tokens', 'reused_tokens'),
                           ('stored_tokens', 'stored_tokens'), ('reason', 'cache_reason'),
                           ('retention_reason', 'retention_reason'), ('prefix_digest', 'cached_prefix_sha256')):
        if source in cache:
            measurement[target] = cache[source]


async def chunks(stream, identity, measurement=None, request_started=None):
    measurement = {} if measurement is None else measurement
    def chunk(delta, finish_reason=None, cache=None):
        return event_frame({**({"cache": cache} if cache is not None else {}), **identity, 'object': 'chat.completion.chunk',
            'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}]})
    yield chunk({'role': 'assistant', 'content': ''})
    finish = None
    cache = None
    try:
        async for event in stream:
            observe_generation(measurement, event)
            if event.get('content') and request_started is not None and 'stream_ttft_ms' not in measurement:
                measurement['stream_ttft_ms'] = (time.monotonic() - request_started) * 1000
            if 'content' in event:
                yield chunk({'content': event['content']})
            if 'cache' in event:
                cache = event['cache']
            if 'finish_reason' in event:
                finish = event['finish_reason']
        if finish not in ('stop', 'length'):
            raise InferenceFailure('Generation ended without a terminal event.', 502)
        yield chunk({}, finish, cache)
        yield 'data: [DONE]\n\n'
    except InferenceFailure as exc:
        yield event_frame({'error': {'message': str(exc), 'type': 'inference_error', 'code': exc.status_code}})
        yield 'data: [DONE]\n\n'


@router.post('', operation_id='createCompletion', response_model=CompletionResponse,
    responses={200: {'content': {'text/event-stream': {'schema': {'type': 'string'}}}}})
async def create_completion(body: CompletionRequest, request: Request):
    """Queue JSON or incremental SSE chat; disconnect waits for native cleanup."""
    measurement = request.scope.setdefault('kadan_measurement', {})
    if body.conversation_id:
        measurement['conversation_key'] = hashlib.sha256(body.conversation_id.encode()).hexdigest()[:16]
    identity = {'id': 'chatcmpl-' + uuid4().hex, 'created': int(time.time()),
                'model': body.model or memory_manager.runtime.model_id or 'unknown'}
    if body.stream:
        try:
            stream, model = await infer(request, memory_manager.open_chat_stream(body))
        except HTTPException:
            raise
        identity['model'] = model or identity['model']
        return OwnedStreamResponse(chunks(stream, identity, measurement,
            request.scope.get('kadan_request_started', time.monotonic())), stream)
    result = await infer(request, memory_manager.submit(body, feature='llm', operation='completion'))
    identity['model'] = result['model']
    observe_generation(measurement, result)
    text = result['text']
    return CompletionResponse(**identity, message=ChatMessage(role='assistant', text=text),
        choices=[CompletionChoice(message=AssistantMessage(content=text), finish_reason=result['finish_reason'])],
        cache=result.get('cache'))
