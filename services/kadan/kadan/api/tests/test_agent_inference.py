import asyncio
import json
from uuid import UUID

import httpx
import pytest

from kadan.api.agent_inference import (
    MAX_COMMAND_CHARACTERS,
    AgentInference,
    Finish,
    GenerateImage,
    SendMessage,
    parse_command,
)
from kadan.api.config import ServiceError
from kadan.api.pydantic_models.inference import CompletionMessage


def chunk(content="", finish=None):
    return (
        "data: "
        + json.dumps(
            {"choices": [{"delta": {"content": content}, "finish_reason": finish}]}
        )
        + "\n\n"
    )


def completed(content):
    return chunk(content) + chunk(finish="stop") + "data: [DONE]\n\n"


def invoke(stream, status=200, media_type="text/event-stream"):
    async def call():
        def respond(request):
            body = json.loads(request.content)
            assert body == {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "selected-model",
                "stream": True,
                "conversation_id": "kadan:test:conversation",
            }
            assert request.extensions["timeout"]["read"] is None
            return httpx.Response(
                status, text=stream, headers={"content-type": media_type}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await AgentInference(client, "http://hikari").call(
                [CompletionMessage(role="user", content="Hello")],
                "selected-model",
                "kadan:test:conversation",
            )

    return asyncio.run(call())


@pytest.mark.parametrize(
    "command,expected",
    [
        (
            {"tool": "send_message", "content": "Hello", "image_job_id": None},
            SendMessage,
        ),
        (
            {
                "tool": "send_message",
                "content": "Here it is",
                "image_job_id": "10000000-0000-4000-8000-000000000001",
            },
            SendMessage,
        ),
        ({"tool": "generate_image", "prompt": "A little blue bird"}, GenerateImage),
        ({"tool": "finish"}, Finish),
    ],
)
def test_valid_commands(command, expected):
    result = invoke(completed(json.dumps(command)))
    assert isinstance(result, expected)
    if command.get("image_job_id"):
        assert result.image_job_id == UUID(command["image_job_id"])


def test_command_can_span_multiple_deltas():
    result = invoke(
        chunk('{"tool":')
        + chunk('"finish"}')
        + chunk(finish="stop")
        + "data: [DONE]\n\n"
    )
    assert isinstance(result, Finish)


def test_saved_commands_use_the_same_validation():
    assert isinstance(parse_command('{"tool":"finish"}'), Finish)
    with pytest.raises(ServiceError) as error:
        parse_command(" " * (MAX_COMMAND_CHARACTERS + 1))
    assert error.value.code == "command_too_large"


@pytest.mark.parametrize(
    "content",
    [
        "Hello there",
        '```json\n{"tool":"finish"}\n```',
        '{"tool":"finish"} trailing text',
        '[{"tool":"finish"}]',
        '{"tool":"finish","content":"duplicate answer"}',
        '{"tool":"send_message","content":" "}',
        '{"tool":"send_message","content":1}',
        '{"tool":"send_message","content":"Here","image_job_id":"wrong"}',
        '{"tool":"generate_image","prompt":""}',
        '{"tool":"unknown"}',
        '{"tool":"finish","tool":"send_message","content":"hi"}',
        '{"tool":"finish"',
        "",
    ],
)
def test_malformed_command_is_not_executed(content):
    with pytest.raises(ServiceError) as error:
        invoke(completed(content))
    assert error.value.code == "invalid_command"


@pytest.mark.parametrize(
    "ending",
    [
        "",
        chunk(finish="stop"),
        "data: [DONE]\n\n",
        chunk(finish="stop") + "data: [DONE]",
        chunk(finish="length") + "data: [DONE]\n\n",
        chunk(finish="tool_calls") + "data: [DONE]\n\n",
        chunk(finish="stop") + chunk("more") + "data: [DONE]\n\n",
        'data: {"choices": []}\n\n',
        'data: {"error":{"message":"failed"}}\n\n',
        'data: {"choices":[{"delta":{"tool_calls":[]}}]}\n\n',
    ],
)
def test_incomplete_or_invalid_stream_cannot_execute_even_a_valid_command(ending):
    with pytest.raises(ServiceError) as error:
        invoke(chunk('{"tool":"finish"}') + ending)
    assert error.value.code in {"incomplete_stream", "invalid_stream"}


def test_command_size_is_bounded_across_deltas():
    with pytest.raises(ServiceError) as error:
        invoke(chunk(" " * MAX_COMMAND_CHARACTERS) + chunk(" "))
    assert error.value.code == "command_too_large"


@pytest.mark.parametrize(
    "status,media_type", [(503, "text/event-stream"), (200, "application/json")]
)
def test_bad_http_response_cannot_execute_command(status, media_type):
    with pytest.raises(ServiceError):
        invoke(completed('{"tool":"finish"}'), status, media_type)


def test_cancel_closes_silent_hikari_stream():
    class SilentStream(httpx.AsyncByteStream):
        def __init__(self):
            self.waiting = asyncio.Event()
            self.closed = False

        async def __aiter__(self):
            self.waiting.set()
            await asyncio.Event().wait()
            yield b""

        async def aclose(self):
            self.closed = True

    async def check():
        stream = SilentStream()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200, stream=stream, headers={"content-type": "text/event-stream"}
                )
            )
        ) as client:
            task = asyncio.create_task(
                AgentInference(client, "http://hikari").call(
                    [CompletionMessage(role="user", content="Hello")],
                    "model",
                    "conversation",
                )
            )
            await stream.waiting.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert stream.closed

    asyncio.run(check())
