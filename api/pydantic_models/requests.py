from typing import Annotated, Literal
from pydantic import BaseModel, Field

RequestType = Literal['LLM', 'Image', 'Video', 'TTS', 'STT', 'Decision']
RequestStatus = Annotated[int, Field(ge=100, le=599)]


class RequestRecord(BaseModel):
    """Completed HTTP-handler observation retained only in the current API process.

    The legacy prompt/output fields contain structural and HTTP summaries, not
    user text or model output. Latency includes validation and cleanup; it is
    not a token throughput or time-to-first-token measurement."""
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
