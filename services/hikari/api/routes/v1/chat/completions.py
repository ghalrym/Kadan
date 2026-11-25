import hashlib
import json
import time
from typing import Literal
from uuid import uuid4

import anyio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from api.pydantic_models.chat import ChatMessage, ChatTool
from api.pydantic_models.inference_task import LlmModel
from api.repositories import subscriptions
from api.services.inference import inference
from api.services.inference_errors import InferenceFailure
from api.services.inference_http import infer
from api.services.subscription_inference import subscription_stream

router = APIRouter(prefix="/v1/chat/completions", tags=["Chat"])


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[ChatMessage] = Field(min_length=1)
    tools: list[ChatTool] | None = Field(default=None, max_length=256)
    model: LlmModel | None = None
    stream: bool = False
    conversation_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Client-owned opaque conversation ID; omitted IDs never retain prefix state.",
    )
    reuse_prefix: bool = Field(
        default=True, description="Disable to compare against full prefill."
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        strict=True,
        description="Cap on generated tokens, including reasoning. Omit to allow the remaining context.",
    )
    reasoning_tokens: int | None = Field(
        default=None,
        ge=1,
        strict=True,
        description="Reasoning is closed after this many tokens so the reply keeps its share of max_tokens.",
    )
    reasoning_effort: Literal["low", "medium", "high"] | None = None


class OutputFunction(BaseModel):
    name: str
    arguments: str


class OutputToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: OutputFunction


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str
    tool_calls: list[OutputToolCall] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class CompletionChoice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: Literal["stop", "length", "tool_calls"] = "stop"


class CacheUsage(BaseModel):
    hit: bool
    reused_tokens: int
    stored_tokens: int
    host_bytes: int
    device_bytes: dict[str, int]
    reason: str
    retention_reason: str = "unknown"
    limit_bytes: int = 0


class CompletionOutputMessage(BaseModel):
    role: Literal["assistant"]
    text: str
    meta: str | None = None

    tool_calls: list[OutputToolCall] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class CompletionResponse(BaseModel):
    # Preserve the original Kadan response while adding the standard envelope.
    message: CompletionOutputMessage
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    cache: CacheUsage | None = None


class OwnedStreamResponse(StreamingResponse):
    def __init__(self, content, owner):
        super().__init__(
            content,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
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
    return "data: " + json.dumps(value, ensure_ascii=False) + "\n\n"


def observe_generation(measurement, event):
    """Copy numeric native measurements and safe cache metadata; never output text."""
    timing = event.get("timing") or {}
    for key in (
        "generation_ttft_ms",
        "prefill_ms",
        "decode_tokens_per_second",
        "output_tokens",
        "prefill_tokens",
    ):
        if key in timing:
            measurement[key] = timing[key]
    cache = event.get("cache") or {}
    for source, target in (
        ("hit", "cache_hit"),
        ("reused_tokens", "reused_tokens"),
        ("stored_tokens", "stored_tokens"),
        ("reason", "cache_reason"),
        ("retention_reason", "retention_reason"),
        ("prefix_digest", "cached_prefix_sha256"),
    ):
        if source in cache:
            measurement[target] = cache[source]


async def chunks(stream, identity, measurement=None, request_started=None):
    measurement = {} if measurement is None else measurement

    def chunk(delta, finish_reason=None, cache=None):
        return event_frame(
            {
                **({"cache": cache} if cache is not None else {}),
                **identity,
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish_reason}
                ],
            }
        )

    yield chunk({"role": "assistant", "content": ""})
    finish = None
    cache = None
    next_call = 0
    try:
        async for event in stream:
            observe_generation(measurement, event)
            if (
                event.get("content")
                and request_started is not None
                and "stream_ttft_ms" not in measurement
            ):
                measurement["stream_ttft_ms"] = (
                    time.monotonic() - request_started
                ) * 1000
            if "content" in event:
                yield chunk({"content": event["content"]})
            if event.get("tool_calls"):
                calls = [
                    {**call, "index": next_call + i}
                    for i, call in enumerate(event["tool_calls"])
                ]
                next_call += len(calls)
                yield chunk({"tool_calls": calls})
            if "cache" in event:
                cache = event["cache"]
            if "finish_reason" in event:
                finish = event["finish_reason"]
        if finish not in ("stop", "length", "tool_calls"):
            raise InferenceFailure("Generation ended without a terminal event.", 502)
        yield chunk({}, finish, cache)
        yield "data: [DONE]\n\n"
    except InferenceFailure as exc:
        yield event_frame(
            {
                "error": {
                    "message": str(exc),
                    "type": "inference_error",
                    "code": exc.status_code,
                }
            }
        )
        yield "data: [DONE]\n\n"


@router.post(
    "",
    operation_id="createCompletion",
    response_model=CompletionResponse,
    responses={200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}}},
)
async def create_completion(body: CompletionRequest, request: Request):
    """Queue JSON or incremental SSE chat; disconnect waits for native cleanup."""
    measurement = request.scope.setdefault("kadan_measurement", {})
    if body.conversation_id:
        measurement["conversation_key"] = hashlib.sha256(
            body.conversation_id.encode()
        ).hexdigest()[:16]
    identity = {
        "id": "chatcmpl-" + uuid4().hex,
        "created": int(time.time()),
        "model": body.model or inference.model_id or "unknown",
    }
    try:
        subscription = await anyio.to_thread.run_sync(subscriptions.active, "language")
    except subscriptions.SubscriptionError as exc:
        raise HTTPException(409, str(exc)) from exc
    if subscription is not None:
        return await subscription_completion(body, request, subscription, identity)
    if body.stream:
        stream, model = await infer(request, inference.open_chat_stream(body))
        identity["model"] = model or identity["model"]
        return OwnedStreamResponse(
            chunks(
                stream,
                identity,
                measurement,
                request.scope.get("kadan_request_started", time.monotonic()),
            ),
            stream,
        )
    result = await infer(
        request, inference.submit(body, feature="llm", operation="completion")
    )
    identity["model"] = result["model"]
    observe_generation(measurement, result)
    return completion_response(identity, result)


def completion_response(identity, result):
    text = result["text"]
    return CompletionResponse(
        **identity,
        message=CompletionOutputMessage(
            role="assistant", text=text, tool_calls=result.get("tool_calls") or None
        ),
        choices=[
            CompletionChoice(
                message=AssistantMessage(
                    content=text, tool_calls=result.get("tool_calls") or None
                ),
                finish_reason=result["finish_reason"],
            )
        ],
        cache=result.get("cache"),
    )


async def subscription_result(stream):
    content, calls, finish = [], [], None
    async for event in stream:
        if "content" in event:
            content.append(event["content"])
        calls.extend(event.get("tool_calls", ()))
        if "finish_reason" in event:
            finish = event["finish_reason"]
    if finish not in ("stop", "length", "tool_calls"):
        raise InferenceFailure("Generation ended without a terminal event.", 502)
    return {"text": "".join(content), "tool_calls": calls, "finish_reason": finish}


async def subscription_completion(body, request, subscription, identity):
    """Serve the turn from the user's subscription in the same response shapes."""
    stream = subscription_stream(subscription, body)
    identity["model"] = stream.model
    if body.stream:
        try:
            await stream.start()
        except InferenceFailure as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        return OwnedStreamResponse(chunks(stream, identity), stream)
    result = await infer(request, subscription_result(stream))
    return completion_response(identity, result)
