import asyncio
import json

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from kadan.api.config import ServiceError
from kadan.runner.agent_inference import (
    MAX_COMMAND_CHARACTERS,
    TOOLS_FOOTER,
    TOOLS_HEADER,
    AgentInference,
    HikariChatModel,
    ToolCommand,
    parse_command,
    parse_tool_calls,
    render_tool_call,
)
from kadan.runner.tools import CONVERSATION_TOOLS
from kadan.runner.tools.shell import shell_exec_command

TOOLS = [*CONVERSATION_TOOLS, shell_exec_command]


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


def call(name, **arguments):
    return render_tool_call(name, arguments)


def invoke(stream, status=200, media_type="text/event-stream", seen=None):
    async def run():
        def respond(request):
            body = json.loads(request.content)
            if seen is not None:
                seen.append(body)
            assert body["model"] == "selected-model"
            assert body["stream"] is True
            assert body["conversation_id"] == "kadan:test:conversation"
            assert request.extensions["timeout"]["read"] is None
            return httpx.Response(
                status, text=stream, headers={"content-type": media_type}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await AgentInference(client, "http://hikari").call(
                [SystemMessage(content="Instructions"), HumanMessage(content="Hello")],
                "selected-model",
                "kadan:test:conversation",
                TOOLS,
            )

    return asyncio.run(run())


def test_every_bound_tool_is_rendered_in_the_checkpoint_tool_format():
    seen = []
    invoke(completed(call("finish")), seen=seen)
    system = seen[0]["messages"][0]
    assert system["role"] == "system"
    assert system["content"].startswith(TOOLS_HEADER)
    assert TOOLS_FOOTER + "\n\nInstructions" in system["content"]
    rendered = [
        json.loads(line)["function"]["name"]
        for line in system["content"].split("\n")
        if line.startswith('{"type": "function"')
    ]
    assert rendered == [tool.name for tool in TOOLS]
    assert seen[0]["messages"][1] == {"role": "user", "content": "Hello"}


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (
            call("send_message", content="Hello", end_run=True),
            ToolCommand(
                tool="send_message", arguments={"content": "Hello", "end_run": True}
            ),
        ),
        (
            call("generate_image", prompt="A little blue bird"),
            ToolCommand(
                tool="generate_image", arguments={"prompt": "A little blue bird"}
            ),
        ),
        (call("finish"), ToolCommand(tool="finish")),
        (
            "I will list the files.\n\n"
            + call("shell_exec_command", command="ls -la", timeout=30),
            ToolCommand(
                tool="shell_exec_command",
                arguments={"command": "ls -la", "timeout": 30},
            ),
        ),
    ],
)
def test_tool_calls_are_parsed_with_schema_typed_arguments(reply, expected):
    assert invoke(completed(reply)) == expected


def test_string_parameters_stay_strings_even_when_they_look_like_json():
    reply = call("shell_exec_command", command="42")
    assert invoke(completed(reply)).arguments == {"command": "42"}


def test_multiline_parameter_values_are_preserved():
    text, calls = parse_tool_calls(
        "<tool_call>\n<function=send_message>\n<parameter=content>\nline one\n"
        "line two\n</parameter>\n<parameter=end_run>\ntrue\n</parameter>\n"
        "</function>\n</tool_call>",
        [{"function": {"name": tool.name}} for tool in TOOLS],
    )
    assert text == ""
    assert calls[0]["args"] == {"content": "line one\nline two", "end_run": True}


def test_calls_without_the_tool_call_wrapper_are_parsed():
    text, calls = parse_tool_calls(
        "\n<function=send_message>\n<parameter=content>\nHello!\n</parameter>\n"
        "<parameter=end_run>\ntrue\n</parameter>\n</function>\n",
        [{"function": {"name": tool.name}} for tool in TOOLS],
    )
    assert text == ""
    assert calls[0]["args"] == {"content": "Hello!", "end_run": True}


def test_only_the_first_of_several_calls_is_taken():
    reply = call("finish") + "\n" + call("send_message", content="x", end_run=True)
    assert invoke(completed(reply)) == ToolCommand(tool="finish")


@pytest.mark.parametrize(
    "reply",
    [
        "Hello there",
        '{"tool":"finish"}',
        "<tool_call>\n<function=finish>\n</tool_call>",
        "<tool_call>\n<function=bad name!>\n</function>\n</tool_call>",
        "",
    ],
)
def test_replies_without_a_tool_call_finish_the_run(reply):
    assert invoke(completed(reply)) == ToolCommand(tool="finish")


def test_saved_commands_round_trip_and_accept_older_formats():
    command = ToolCommand(tool="send_message", arguments={"content": "Hi"})
    assert parse_command(command.model_dump_json()) == command
    assert parse_command('{"tool":"finish"}') == ToolCommand(tool="finish")
    assert parse_command(
        '{"tool":"send_message","content":"Hi","end_run":true}'
    ) == ToolCommand(tool="send_message", arguments={"content": "Hi", "end_run": True})
    assert parse_command(
        '{"tool":"computer","operation":"shell.exec_command","arguments":{"command":"ls"}}'
    ) == ToolCommand(tool="shell_exec_command", arguments={"command": "ls"})
    with pytest.raises(ServiceError) as error:
        parse_command(" " * (MAX_COMMAND_CHARACTERS + 1))
    assert error.value.code == "command_too_large"


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
        invoke(chunk(call("finish")) + ending)
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
        invoke(completed(call("finish")), status, media_type)


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
                    [HumanMessage(content="Hello")], "model", "conversation", TOOLS
                )
            )
            await stream.waiting.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert stream.closed

    asyncio.run(check())


def test_tool_calls_and_results_replay_as_checkpoint_text():
    wire = HikariChatModel._wire_messages(
        [
            SystemMessage(content="Instructions"),
            HumanMessage(content="Hello"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "shell_exec_command", "args": {"command": "ls"}, "id": "1"}
                ],
            ),
            ToolMessage(content='{"output": "a.txt"}', tool_call_id="1"),
            HumanMessage(content="Thanks"),
        ],
        [],
    )
    assert wire == [
        {"role": "system", "content": "Instructions"},
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "<tool_call>\n<function=shell_exec_command>\n"
            "<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>",
        },
        {
            "role": "user",
            "content": '<tool_response>\n{"output": "a.txt"}\n</tool_response>',
        },
        {"role": "user", "content": "Thanks"},
    ]


def test_hikari_chat_model_rejects_stop_sequences():
    async def check():
        async with httpx.AsyncClient() as client:
            model = HikariChatModel(
                client=client,
                hikari_url="http://hikari",
                model_id="model",
                conversation_key="conversation",
            )
            with pytest.raises(ValueError, match="does not support"):
                await model.ainvoke([HumanMessage(content="Hello")], stop=["end"])

    asyncio.run(check())


def test_hikari_chat_model_rejects_non_text_messages():
    with pytest.raises(TypeError):
        HikariChatModel._wire_messages(
            [HumanMessage(content=[{"type": "text", "text": "Hello"}])], []
        )
