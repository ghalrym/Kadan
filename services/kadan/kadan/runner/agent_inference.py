"""Hikari chat with LangChain tool binding over its text-only stream.

Native Hikari accepts only text messages, so `bind_tools` renders the tool definitions
into the system message in the Qwen checkpoint's own tool format and parses
`<tool_call>` blocks in the reply back into LangChain tool calls.
The explicit LFM3 trial instead sends structured tools and tool-role history to
its checkpoint's own chat template; the same parser and action owner apply.
"""

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
from pydantic import Field, ValidationError

from ..api.config import ServiceError
from ..api.pydantic_models.agent_runs import ToolCommand

MAX_COMMAND_CHARACTERS = 48000
# Output caps sent to Hikari: reasoning is closed after its budget, and the reply
# budget covers the largest command the runner accepts (three characters per token).
REASONING_TOKENS = 16384
REPLY_TOKENS = MAX_COMMAND_CHARACTERS // 3
MAX_EVENT_CHARACTERS = 300000
INFERENCE_TIMEOUT = httpx.Timeout(None, connect=3)
logger = logging.getLogger(__name__)

# The checkpoint chat template's tool section, verbatim.
TOOLS_HEADER = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
TOOLS_FOOTER = (
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following "
    "format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n"
    "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
    "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an "
    "inner <function=...></function> block must be nested within "
    "<tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n- If there is no function "
    "call available, answer the question like normal with your current knowledge "
    "and do not tell the user about function calls\n</IMPORTANT>"
)
# Hikari drops special tokens, so the <tool_call> wrapper may not arrive.
TOOL_CALL = re.compile(
    r"(?:<tool_call>\s*)?<function=([^>\s]+)>(.*?)</function>(?:\s*</tool_call>)?",
    re.DOTALL,
)
PARAMETER = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
TOOL_MARKUP = re.compile(
    r"</?(?:tool_call|function|parameter|tool_response)(?:[=>\s]|$)", re.IGNORECASE
)


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


def render_tool_call(name: str, arguments: dict[str, Any]) -> str:
    parameters = "".join(
        f"<parameter={key}>\n"
        + (value if isinstance(value, str) else json.dumps(value))
        + "\n</parameter>\n"
        for key, value in arguments.items()
    )
    return f"<tool_call>\n<function={name}>\n{parameters}</function>\n</tool_call>"


def is_string(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    options = schema.get("anyOf") or schema.get("oneOf")
    if options:
        kinds = {option.get("type") for option in options if isinstance(option, dict)}
        return kinds - {"null"} == {"string"}
    return schema.get("type") == "string"


def tool_section(tools: Sequence[dict]) -> str:
    """The system prompt's tool definitions, as the checkpoint template renders them."""
    return (
        TOOLS_HEADER + "".join("\n" + json.dumps(tool) for tool in tools) + TOOLS_FOOTER
    )


def tool_characters(tools: Sequence[BaseTool]) -> int:
    """Characters the bound tools add to the prompt."""
    return len(tool_section([convert_to_openai_tool(tool) for tool in tools]))


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
    """Hikari's text-only API with tools rendered in the checkpoint's format."""

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
    def _wire_messages(
        messages: list[BaseMessage], tools: Sequence[dict]
    ) -> list[dict[str, str]]:
        wire: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message.content, str):
                raise TypeError("Hikari accepts only text messages")
            if isinstance(message, ToolMessage):
                response = f"<tool_response>\n{message.content}\n</tool_response>"
                if (
                    wire
                    and wire[-1]["role"] == "user"
                    and wire[-1]["content"].startswith("<tool_response>")
                ):
                    wire[-1]["content"] += "\n" + response
                else:
                    wire.append({"role": "user", "content": response})
            elif isinstance(message, AIMessage):
                if message.invalid_tool_calls:
                    raise ValueError("Hikari cannot replay invalid tool calls")
                content = "\n\n".join(
                    part
                    for part in [
                        message.content.strip(),
                        *(
                            render_tool_call(call["name"], call["args"])
                            for call in message.tool_calls
                        ),
                    ]
                    if part
                )
                wire.append({"role": "assistant", "content": content})
            elif isinstance(message, SystemMessage):
                wire.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                wire.append({"role": "user", "content": message.content})
            else:
                raise TypeError(
                    "Hikari accepts system, user, assistant and tool messages"
                )
        if tools:
            section = tool_section(tools)
            if wire and wire[0]["role"] == "system":
                wire[0]["content"] = section + "\n\n" + wire[0]["content"].strip()
            else:
                wire.insert(0, {"role": "system", "content": section})
        return wire

    @staticmethod
    def _lfm3_wire_messages(messages: list[BaseMessage]) -> list[dict]:
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
        experimental = self.model_id == "lfm3-1.7b-bf16"
        try:
            async with self.client.stream(
                "POST",
                self.hikari_url + "/v1/chat/completions",
                json={
                    "messages": self._lfm3_wire_messages(messages)
                    if experimental
                    else self._wire_messages(messages, tools),
                    **({"tools": list(tools)} if experimental else {}),
                    **(
                        {
                            "max_tokens": REASONING_TOKENS + REPLY_TOKENS,
                            "reasoning_tokens": REASONING_TOKENS,
                        }
                        if experimental
                        else {}
                    ),
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
            if model_id == "lfm3-1.7b-bf16" and response.content.strip():
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
