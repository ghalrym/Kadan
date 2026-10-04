from typing import Literal
from pydantic import BaseModel, Field

RequestType = Literal["LLM", "Image", "Video", "TTS", "STT", "Decision"]
RequestStatus = Literal[200, 202, 429, 500]


class RequestRecord(BaseModel):
    id: str
    time: str
    type: RequestType
    model: str
    status: RequestStatus
    latency: str
    ttft: str | None = None
    tokens_per_second: int | None = Field(default=None, alias="tokensPerSecond")
    endpoint: str
    prompt: str
    output: str
