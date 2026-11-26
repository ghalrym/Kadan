"""Hikari chat with structured tool calls and streamed assistant replies."""

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any, Literal

import httpx
from kadan_tracing import traced
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, ValidationError

from ..api.config import ServiceError
from ..api.pydantic_models.agent_runs import RunCall, ToolCommand

MAX_COMMAND_CHARACTERS = 48000
# Output caps sent to Hikari: reasoning is closed after its budget, and the reply
# budget estimates three characters per token; some valid-size commands may not fit.
REASONING_TOKENS = 16384
REPLY_TOKENS = MAX_COMMAND_CHARACTERS // 3
MAX_EVENT_CHARACTERS = 300000
INFERENCE_TIMEOUT = httpx.Timeout(None, connect=3)
logger = logging.getLogger(__name__)


def unique_fields(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON field")
        value[key] = item
    return value


def tool_characters(tools: Sequence[BaseTool]) -> int:
    """Characters the bound tool definitions add to the prompt budget."""
    serialized = json.dumps(
        [convert_to_openai_tool(tool) for tool in tools],
        separators=(",", ":"),
        sort_keys=True,
    )
    return len(serialized)


def parse_call(value: dict, index: int) -> dict:
    """Hikari emits one complete call per indexed delta after decoding its arguments."""
    try:
        if (
            value["type"] != "function"
            or type(value["index"]) is not int
            or value["index"] != index
            or not isinstance(value["id"], str)
            or not value["id"]
        ):
            raise ValueError("Invalid tool call identity")
        function = value["function"]
        arguments = json.loads(function["arguments"], object_pairs_hook=unique_fields)
        command = ToolCommand(tool=function["name"], arguments=arguments)
        return {"name": command.tool, "args": command.arguments, "id": value["id"]}
    except (KeyError, TypeError, ValueError) as error:
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
    """Hikari's structured chat completions API with streamed tool calls."""

    client: httpx.AsyncClient = Field(exclude=True)
    hikari_url: str
    model_id: str | None = None
    conversation_key: str
    priority: Literal["interactive", "background"] = "interactive"
    on_text_delta: Callable[[str], Awaitable[None]] | None = Field(
        default=None, exclude=True
    )

    @property
    def _llm_type(self) -> str:
        return "hikari"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return super().bind(
            tools=[convert_to_openai_tool(tool) for tool in tools], **kwargs
        )

    @staticmethod
    def _wire_messages(messages: list[BaseMessage]) -> list[dict]:
        wire = []
        for message in messages:
            if isinstance(message.content, list) and any(
                not isinstance(part, dict)
                or part.get("type") not in {"text", "image_url"}
                for part in message.content
            ):
                raise TypeError("Hikari expects ordered text and image content parts")
            if isinstance(message, ToolMessage):
                wire.append(
                    {
                        "role": "tool",
                        "content": message.content,
                        "tool_call_id": message.tool_call_id,
                        **({"name": message.name} if message.name else {}),
                    }
                )
            elif isinstance(message, AIMessage):
                if message.invalid_tool_calls:
                    raise ValueError("Hikari cannot replay invalid tool calls")
                row: dict[str, Any] = {"role": "assistant", "content": message.content}
                if message.tool_calls:
                    row["tool_calls"] = [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call["args"],
                            },
                        }
                        for call in message.tool_calls
                    ]
                wire.append(row)
            elif isinstance(message, SystemMessage):
                wire.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                wire.append({"role": "user", "content": message.content})
            else:
                raise TypeError(
                    "Hikari accepts system, user, assistant and tool messages"
                )
        return wire

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
        tools = kwargs.pop("tools", [])
        if stop is not None or kwargs:
            raise ValueError("Hikari does not support stop sequences or model options")
        content: list[str] = []
        calls: list[dict] = []
        length = 0
        finished = False
        done = False
        payload: dict[str, Any] = {
            "messages": self._wire_messages(messages),
            "tools": list(tools),
            "max_tokens": REASONING_TOKENS + REPLY_TOKENS,
            "reasoning_tokens": REASONING_TOKENS,
            "stream": True,
            "conversation_id": self.conversation_key,
            "priority": self.priority,
        }
        if self.model_id is not None:
            payload["model"] = self.model_id
        try:
            async with self.client.stream(
                "POST",
                self.hikari_url + "/v1/chat/completions",
                json=payload,
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
                            "tool_calls",
                        }:
                            raise ValueError("Expected text delta")
                        text = delta.get("content", "")
                        reason = choice.get("finish_reason")
                        if not isinstance(text, str) or finished:
                            raise ValueError("Invalid or post-terminal delta")
                        if "tool_calls" in delta:
                            raw_calls = delta["tool_calls"]
                            if not isinstance(raw_calls, list) or not raw_calls:
                                raise ValueError("Empty tool call delta")
                            for raw_call in raw_calls:
                                length += len(json.dumps(raw_call, ensure_ascii=False))
                                calls.append(parse_call(raw_call, len(calls)))
                        if reason in {"stop", "tool_calls"} and bool(calls) != (
                            reason == "tool_calls"
                        ):
                            raise ValueError("Tool calls do not match terminal reason")
                        if reason not in (None, "stop", "length", "tool_calls"):
                            raise ValueError("Incomplete inference")
                    except (ValueError, KeyError, TypeError) as error:
                        raise ServiceError(
                            "Hikari returned an invalid or incomplete agent response.",
                            502,
                            "invalid_stream",
                        ) from error
                    if reason == "length":
                        raise ServiceError(
                            "Hikari reached the model's output limit before completing the agent response.",
                            502,
                            "response_limit",
                        )
                    length += len(text)
                    if length > MAX_COMMAND_CHARACTERS:
                        raise ServiceError(
                            "The agent command exceeded its size limit.",
                            502,
                            "command_too_large",
                        )
                    content.append(text)
                    if text and not calls and self.on_text_delta is not None:
                        await self.on_text_delta(text)
                    finished = reason in ("stop", "tool_calls")
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
            generations=[
                ChatGeneration(
                    message=AIMessage(content="".join(content), tool_calls=calls)
                )
            ]
        )


class AgentInference:
    def __init__(self, client: httpx.AsyncClient, hikari_url: str):
        self.client = client
        self.hikari_url = hikari_url

    @traced("kadan.agent_inference.call")
    async def call(
        self,
        messages: list[BaseMessage],
        model_id: str | None,
        conversation_key: str,
        tools: Sequence[BaseTool],
        *,
        priority: Literal["interactive", "background"] = "interactive",
        on_text_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> RunCall:
        model = HikariChatModel(
            client=self.client,
            hikari_url=self.hikari_url,
            model_id=model_id,
            conversation_key=conversation_key,
            priority=priority,
            on_text_delta=on_text_delta,
        ).bind_tools(tools)
        response = await model.ainvoke(messages)
        if not response.tool_calls:
            if response.content.strip() and any(
                tool.name == "send_message" for tool in tools
            ):
                return RunCall(
                    command=ToolCommand(
                        tool="send_message",
                        arguments={"content": response.content, "end_run": True},
                    )
                )
            # Not responding is allowed: an empty reply or unbound plain reply ends the run.
            logger.info("Agent reply had no tool call: %r", response.content[:2000])
            return RunCall(command=ToolCommand(tool="finish"))
        if len(response.tool_calls) != 1:
            raise ServiceError(
                "The agent requested multiple actions in a single step.",
                502,
                "invalid_command",
            )
        call = response.tool_calls[0]
        try:
            return RunCall(
                call_id=call["id"],
                command=ToolCommand(tool=call["name"], arguments=call["args"]),
            )
        except ValidationError as error:
            raise ServiceError(
                "The agent returned an invalid tool command.", 502, "invalid_command"
            ) from error
