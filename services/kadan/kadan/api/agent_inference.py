"""The agent's small command protocol over Hikari's text-only chat stream."""

import json
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal
from uuid import UUID

import httpx
from kadan_tracing import traced
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from .config import ServiceError
from .pydantic_models.inference import CompletionMessage

MAX_COMMAND_CHARACTERS = 48000
MAX_EVENT_CHARACTERS = 300000
INFERENCE_TIMEOUT = httpx.Timeout(None, connect=3)

TOOL_PROTOCOL = """You control this conversation through exactly one JSON command per response.
Return only a JSON object, without Markdown, reasoning, or text outside the object.
The available commands are:
{"tool":"send_message","content":"Text to show the user","image_job_id":null,"end_run":true}
{"tool":"generate_image","prompt":"Image description"}
{"tool":"finish"}
send_message delivers visible text immediately. To deliver a completed image, set
image_job_id to its job UUID. Only deliver images whose completion event you received.
Set end_run=true when this message completes the immediate work. This delivers the
message and ends the run without another inference call. Use false only when you
need another tool now.
generate_image saves a background job and its ID, then ends this run immediately;
submission is not delivery. Submit directly without a separate acknowledgment.
Tool results and image completion or failure events arrive as user-role messages.
finish ends this run without sending text. Image events will wake you again.
Use precisely the documented fields. Content must be nonblank and at most 16000
characters; an image prompt must be nonblank and at most 8000 characters.
"""


class SendMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["send_message"]
    content: str = Field(min_length=1, max_length=16000, pattern=r"\S")
    image_job_id: UUID | None = None
    end_run: bool = Field(strict=True)


class GenerateImage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["generate_image"]
    prompt: str = Field(min_length=1, max_length=8000, pattern=r"\S")


class Finish(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["finish"]


AgentCommand = Annotated[
    SendMessage | GenerateImage | Finish, Field(discriminator="tool")
]
COMMAND = TypeAdapter(AgentCommand)


def unique_fields(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON field")
        value[key] = item
    return value


def parse_command(text: str) -> AgentCommand:
    if len(text) > MAX_COMMAND_CHARACTERS:
        raise ServiceError(
            "The agent command exceeded its size limit.", 502, "command_too_large"
        )
    try:
        value = json.loads(text, object_pairs_hook=unique_fields)
        return COMMAND.validate_python(value)
    except (ValueError, ValidationError) as error:
        raise ServiceError(
            "The agent returned an invalid tool command.", 502, "invalid_command"
        ) from error


async def stream_data(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    data: list[str] = []
    size = 0
    async for line in lines:
        if line == "":
            if data:
                yield "\n".join(data)
            data, size = [], 0
        elif line.startswith("data:"):
            value = line[5:].removeprefix(" ")
            size += len(value)
            if size > MAX_EVENT_CHARACTERS:
                raise ServiceError(
                    "Hikari returned an oversized event.", 502, "invalid_stream"
                )
            data.append(value)
    # An unterminated SSE frame is incomplete and must never execute a command.


class HikariChatModel(BaseChatModel):
    """Hikari's text-only API; tools remain in the agent's JSON protocol."""

    client: httpx.AsyncClient = Field(exclude=True)
    hikari_url: str
    model_id: str
    conversation_key: str

    @property
    def _llm_type(self) -> str:
        return "hikari"

    @staticmethod
    def _wire_message(message: BaseMessage) -> dict[str, str]:
        roles = {SystemMessage: "system", HumanMessage: "user", AIMessage: "assistant"}
        role = roles.get(type(message))
        if role is None or not isinstance(message.content, str):
            raise ValueError(
                "Hikari accepts only system, user and assistant text messages"
            )
        if isinstance(message, AIMessage) and (
            message.tool_calls or message.invalid_tool_calls
        ):
            raise ValueError("Hikari does not support native tool calls")
        return {"role": role, "content": message.content}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError(
            "HikariChatModel supports asynchronous inference only"
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stop is not None or kwargs:
            raise ValueError("Hikari does not support stop sequences or model options")
        content: list[str] = []
        length = 0
        finished = False
        done = False
        try:
            async with self.client.stream(
                "POST",
                self.hikari_url + "/v1/chat/completions",
                json={
                    "messages": [self._wire_message(message) for message in messages],
                    "model": self.model_id,
                    "stream": True,
                    "conversation_id": self.conversation_key,
                },
                timeout=INFERENCE_TIMEOUT,
            ) as response:
                if not response.is_success:
                    raise ServiceError(
                        f"Hikari returned HTTP {response.status_code}.",
                        response.status_code,
                        "inference_error",
                    )
                if not response.headers.get("content-type", "").startswith(
                    "text/event-stream"
                ):
                    raise ServiceError(
                        "Hikari did not return a chat stream.", 502, "invalid_stream"
                    )
                async for data in stream_data(response.aiter_lines()):
                    if data == "[DONE]":
                        done = True
                        break
                    try:
                        chunk = json.loads(data, object_pairs_hook=unique_fields)
                        if not isinstance(chunk, dict) or "error" in chunk:
                            raise ValueError("Invalid or failed inference event")
                        choices = chunk["choices"]
                        if not isinstance(choices, list) or len(choices) != 1:
                            raise ValueError("Expected one choice")
                        choice = choices[0]
                        delta = choice["delta"]
                        if not isinstance(delta, dict) or set(delta) - {
                            "role",
                            "content",
                        }:
                            raise ValueError("Expected text delta")
                        text = delta.get("content", "")
                        reason = choice.get("finish_reason")
                        if not isinstance(text, str) or finished:
                            raise ValueError("Invalid or post-terminal delta")
                        if reason is not None and reason != "stop":
                            raise ValueError("Incomplete inference")
                    except (ValueError, KeyError, TypeError) as error:
                        raise ServiceError(
                            "Hikari returned an invalid or incomplete agent response.",
                            502,
                            "invalid_stream",
                        ) from error
                    length += len(text)
                    if length > MAX_COMMAND_CHARACTERS:
                        raise ServiceError(
                            "The agent command exceeded its size limit.",
                            502,
                            "command_too_large",
                        )
                    content.append(text)
                    finished = reason == "stop"
            if not done or not finished:
                raise ServiceError(
                    "Hikari disconnected before completing the command.",
                    502,
                    "incomplete_stream",
                )
        except httpx.RequestError as error:
            raise ServiceError(
                "Hikari disconnected during the agent run.",
                503,
                "inference_unavailable",
            ) from error
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="".join(content)))]
        )


class AgentInference:
    def __init__(self, client: httpx.AsyncClient, hikari_url: str):
        self.client = client
        self.hikari_url = hikari_url

    @traced("kadan.agent_inference.call")
    async def call(
        self,
        messages: list[CompletionMessage],
        model_id: str,
        conversation_key: str,
    ) -> AgentCommand:
        roles = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
        model = HikariChatModel(
            client=self.client,
            hikari_url=self.hikari_url,
            model_id=model_id,
            conversation_key=conversation_key,
        )
        response = await model.ainvoke(
            [roles[message.role](content=message.content) for message in messages]
        )
        return parse_command(response.text)
