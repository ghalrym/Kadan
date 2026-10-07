from typing import Annotated, Literal
from pydantic import BaseModel, Field

RequestType = Literal['LLM', 'Image', 'Video', 'TTS', 'STT', 'Decision']
RequestStatus = Annotated[int, Field(ge=100, le=599)]


class RequestRecord(BaseModel):
    """Completed HTTP-handler observation retained only in the current API process.

    The legacy prompt/output fields contain structural and HTTP summaries, not
    user text or model output. Latency includes validation and cleanup; it is
    not a token throughput or time-to-first-token measurement. Optional native
    timings measure admission/prefill to the first generated token, prefill-only
    work, and decode after that token. Stream TTFT measures HTTP arrival to the
    first content event passed to ASGI (not browser receipt). Missing data is null."""
    id: str
    time: str
    type: RequestType
    model: str | None = None
    status: RequestStatus
    latency: str
    latency_ms: float
    endpoint: str
    prompt: str
    output: str
    request_bytes: int
    response_bytes: int

    generation_ttft_ms: float | None = None
    stream_ttft_ms: float | None = None
    prefill_ms: float | None = None
    decode_tokens_per_second: float | None = None
    output_tokens: int | None = None
    prefill_tokens: int | None = None
    conversation_key: str | None = None
    cache_hit: bool | None = None
    reused_tokens: int | None = None
    stored_tokens: int | None = None
    cache_reason: str | None = None
    retention_reason: str | None = None
    cached_prefix_sha256: str | None = None
