"""Structured LFM3 prompts and validated agent commands over Hikari streams."""

import json
import logging
import re
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

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
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from ..api.config import ServiceError

MAX_COMMAND_CHARACTERS = 48000
MAX_EVENT_CHARACTERS = 300000
INFERENCE_TIMEOUT = httpx.Timeout(None, connect=3)
logger = logging.getLogger(__name__)

TOOL_CALL = re.compile(
    r"(?:<tool_call>\s*)?<function=([^>\s]+)>(.*?)</function>(?:\s*</tool_call>)?",
    re.DOTALL,
)
PARAMETER = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
TOOL_MARKUP = re.compile(
    r"</?(?:tool_call|function|parameter|tool_response)(?:[=>\s]|$)", re.IGNORECASE
)


class ToolCommand(BaseModel):
    """One tool call, persisted in the run transcript before it executes."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    tool: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_]+$")
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


def unique_fields(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON field")
        value[key] = item
    return value


def parse_command(text: str) -> ToolCommand:
    """Read a saved command. Runs saved before tool binding used flat fields."""
    if len(text) > MAX_COMMAND_CHARACTERS:
        raise ServiceError(
            "The agent command exceeded its size limit.", 502, "command_too_large"
        )
    try:
        value = json.loads(text, object_pairs_hook=unique_fields)
        if isinstance(value, dict) and "arguments" not in value:
            value = {
                "tool": value.get("tool"),
                "arguments": {
                    key: item for key, item in value.items() if key != "tool"
                },
            }
        elif isinstance(value, dict) and value.get("tool") == "computer":
            value = {
                "tool": str(value.get("operation", "")).replace(".", "_"),
                "arguments": value.get("arguments", {}),
            }
        return ToolCommand.model_validate(value)
    except (ValueError, ValidationError) as error:
        raise ServiceError(
            "The agent returned an invalid tool command.", 502, "invalid_command"
        ) from error


def is_string(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    options = schema.get("anyOf") or schema.get("oneOf")
    if options:
        kinds = {option.get("type") for option in options if isinstance(option, dict)}
        return kinds - {"null"} == {"string"}
    return schema.get("type") == "string"


def tool_characters(tools: Sequence[BaseTool]) -> int:
    """Characters the bound tools add to the prompt."""
    functions = [convert_to_openai_tool(tool)["function"] for tool in tools]
    return len(
        "<tools>\n"
        + "\n".join(json.dumps(function) for function in functions)
        + "\n</tools>"
    )


def parse_tool_calls(text: str, tools: Sequence[dict]) -> tuple[str, list[dict]]:
    """Split a reply into its text and `<tool_call>` blocks."""
    parameters = {
        tool["function"]["name"]: tool["function"]
        .get("parameters", {})
        .get("properties", {})
        for tool in tools
    }
    calls = []
    for index, match in enumerate(TOOL_CALL.finditer(text)):
        name, body = match.groups()
        properties = parameters.get(name, {})
        arguments: dict[str, Any] = {}
        for key, raw in PARAMETER.findall(body):
            if key in arguments:
                raise ServiceError(
                    "The agent returned an invalid tool command.",
                    502,
                    "invalid_command",
                )
            if is_string(properties.get(key)):
                schema = properties[key]
                options = schema.get("anyOf") or schema.get("oneOf") or []
                nullable = any(option.get("type") == "null" for option in options)
                arguments[key] = None if nullable and raw.strip() == "null" else raw
                continue
            try:
                arguments[key] = json.loads(raw)
            except ValueError:
                arguments[key] = raw
        calls.append({"name": name, "args": arguments, "id": f"call_{index}"})
    first = TOOL_CALL.search(text)
    return (text[: first.start()] if first else text).strip(), calls


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
    """The fixed LFM3 API with checkpoint-rendered tools and structured history."""

    client: httpx.AsyncClient = Field(exclude=True)
    hikari_url: str
    model_id: str
    conversation_key: str

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
            if not isinstance(message.content, str):
                raise TypeError("The LFM3 text trial accepts only text content")
            if isinstance(message, ToolMessage):
                wire.append(
                    {
                        "role": "tool",
                        "content": message.content,
                        "tool_call_id": message.tool_call_id,
                    }
                )
            elif isinstance(message, AIMessage):
                if message.invalid_tool_calls:
                    raise ValueError("Hikari cannot replay invalid tool calls")
                row = {"role": "assistant", "content": message.content}
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
        length = 0
        finished = False
        done = False
        if self.model_id != "lfm3-1.7b-bf16":
            raise ValueError("Kadan uses the fixed LFM3 conversation model")
        try:
            async with self.client.stream(
                "POST",
                self.hikari_url + "/v1/chat/completions",
                json={
                    "messages": self._wire_messages(messages),
                    "tools": list(tools),
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
                        if reason not in (None, "stop", "length"):
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
        text, calls = parse_tool_calls("".join(content), tools)
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=text, tool_calls=calls))
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
        model_id: str,
        conversation_key: str,
        tools: Sequence[BaseTool],
    ) -> ToolCommand:
        model = HikariChatModel(
            client=self.client,
            hikari_url=self.hikari_url,
            model_id=model_id,
            conversation_key=conversation_key,
        ).bind_tools(tools)
        response = await model.ainvoke(messages)
        if not response.tool_calls:
            if response.content.strip():
                if TOOL_MARKUP.search(response.content):
                    raise ServiceError(
                        "The agent returned an invalid tool command.",
                        502,
                        "invalid_command",
                    )
                if any(tool.name == "send_message" for tool in tools):
                    return ToolCommand(
                        tool="send_message",
                        arguments={
                            "content": response.content.strip(),
                            "end_run": True,
                        },
                    )
            # Not responding is allowed: a reply without a tool call ends the run.
            logger.info("Agent reply had no tool call: %r", response.content[:2000])
            return ToolCommand(tool="finish")
        # Runs execute and persist one action at a time; the transcript keeps
        # only this call, so the model never sees later calls as done.
        call = response.tool_calls[0]
        try:
            return ToolCommand(tool=call["name"], arguments=call["args"])
        except ValidationError as error:
            raise ServiceError(
                "The agent returned an invalid tool command.", 502, "invalid_command"
            ) from error
