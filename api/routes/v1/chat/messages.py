from fastapi import APIRouter
from pydantic import BaseModel
from api.pydantic_models.chat import ChatMessage

router = APIRouter(prefix="/v1/chat/messages", tags=["Chat"])

MOCK_MESSAGES = [
    ChatMessage(
        role='user',
        text=('We saw p95 latency jump from 1.1s to 4.8s around 14:02. The request log '
              "shows a burst of image edits from batch-worker. What's the likely "
              'cause?'),
    ),
    ChatMessage(
        role='assistant',
        text=('Most likely GPU contention. Image edits run on the same device as the '
              'LLM, and a burst of them holds the GPU long enough to queue chat '
              'requests behind them.\n'
              '\n'
              'Three things to check:\n'
              '1. Queue time vs. inference time on the slow chat requests. If queue '
              "time dominates, it's contention, not the model.\n"
              '2. Whether batch-worker sets a concurrency limit. Without one it can '
              'take every slot.\n'
              '3. VRAM headroom. If the image model pushed the KV cache out, the LLM '
              'will re-prefill long prompts.\n'
              '\n'
              'A per-client rate limit on /v1/images/edits would stop it happening '
              'again.'),
        meta='214 tokens · 3.1 s',
    ),
]


class MessagesResponse(BaseModel):
    messages: list[ChatMessage]


@router.get("", operation_id="listMessages")
def list_messages() -> MessagesResponse:
    return MessagesResponse(messages=MOCK_MESSAGES)
