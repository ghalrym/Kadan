from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field
from api.pydantic_models.chat import ChatMessage

router = APIRouter(prefix="/v1/chat/completions", tags=["Chat"])

MOCK_COMPLETION = ChatMessage(
    role='assistant',
    text=('Most likely GPU contention. Image edits run on the same device as the LLM, '
          'and a burst of them holds the GPU long enough to queue chat requests behind '
          'them.\n'
          '\n'
          'Three things to check:\n'
          '1. Queue time vs. inference time on the slow chat requests. If queue time '
          "dominates, it's contention, not the model.\n"
          '2. Whether batch-worker sets a concurrency limit. Without one it can take '
          'every slot.\n'
          '3. VRAM headroom. If the image model pushed the KV cache out, the LLM will '
          're-prefill long prompts.\n'
          '\n'
          'A per-client rate limit on /v1/images/edits would stop it happening again.'),
    meta='214 tokens · 3.1 s',
)


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[ChatMessage] = Field(min_length=1)
    model: str = "Qwen3 32B Instruct"


class CompletionResponse(BaseModel):
    message: ChatMessage


@router.post("")
def create_completion(body: CompletionRequest) -> CompletionResponse:
    return CompletionResponse(message=MOCK_COMPLETION)
