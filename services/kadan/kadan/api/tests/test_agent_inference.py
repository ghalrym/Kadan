import asyncio
import json

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError

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
from kadan.runner.tools.file import file_grep_files, file_read_file
from kadan.runner.tools.shell import shell_exec_command

TOOLS = [*CONVERSATION_TOOLS, shell_exec_command, file_read_file]


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


def invoke(
    stream,
    status=200,
    media_type="text/event-stream",
    seen=None,
    model_id="selected-model",
    bound_tools=TOOLS,
    history=None,
):
    async def run():
        def respond(request):
            body = json.loads(request.content)
            if seen is not None:
                seen.append(body)
            assert body["model"] == model_id
            assert body["stream"] is True
            assert body["conversation_id"] == "kadan:test:conversation"
            assert request.extensions["timeout"]["read"] is None
            return httpx.Response(
                status, text=stream, headers={"content-type": media_type}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await AgentInference(client, "http://hikari").call(
                history
                if history is not None
                else [
                    SystemMessage(content="Instructions"),
                    HumanMessage(content="Hello"),
                ],
                model_id,
                "kadan:test:conversation",
                bound_tools,
            )

    return asyncio.run(run())


def test_lfm3_plain_answer_after_tool_uses_existing_reply_action():
    history = [
        SystemMessage(content="Instructions"),
        HumanMessage(content="What is on the screen?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "computer_look",
                    "args": {"question": "Describe the screen"},
                    "id": "call_0",
                }
            ],
        ),
        ToolMessage(content="Google search page", tool_call_id="call_0"),
    ]
    answer = 'One clearly visible word is **"Google"**.'
    assert invoke(
        completed(answer), model_id="lfm3-1.7b-bf16", history=history
    ) == ToolCommand(
        tool="send_message",
        arguments={"content": answer, "end_run": True},
    )


def test_native_plain_answer_still_finishes_without_delivery():
    assert invoke(completed("A plain answer")) == ToolCommand(tool="finish")


@pytest.mark.parametrize("answer", ["", "   ", call("finish")])
def test_lfm3_empty_answer_and_explicit_finish_remain_silent(answer):
    assert invoke(completed(answer), model_id="lfm3-1.7b-bf16") == ToolCommand(
        tool="finish"
    )


def test_lfm3_plain_answer_cannot_create_unbound_reply_action():
    assert invoke(
        completed("A plain answer"),
        model_id="lfm3-1.7b-bf16",
        bound_tools=[shell_exec_command],
    ) == ToolCommand(tool="finish")


@pytest.mark.parametrize(
    "answer",
    [
        "<tool_call>unfinished",
        "<function=send_message>unfinished",
        "<parameter=content>Hello",
        "Answer </tool_call>",
        "<FUNCTION send_message>",
    ],
)
def test_lfm3_malformed_tool_output_is_not_published_as_plain_answer(answer):
    with pytest.raises(ServiceError) as error:
        invoke(completed(answer), model_id="lfm3-1.7b-bf16")
    assert error.value.code == "invalid_command"


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


def test_lfm3_sends_structured_tools_and_true_tool_result_roles():
    seen = []

    async def run():
        def respond(request):
            seen.append(json.loads(request.content))
            return httpx.Response(
                200,
                text=completed(call("finish")),
                headers={"content-type": "text/event-stream"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await AgentInference(client, "http://hikari").call(
                [
                    SystemMessage(content="Owned instructions"),
                    HumanMessage(content="Read it"),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "file_read_file",
                                "args": {"file": "/tmp/a"},
                                "id": "call_0",
                            }
                        ],
                    ),
                    ToolMessage(content="Result", tool_call_id="call_0"),
                ],
                "lfm3-1.7b-bf16",
                "test-conversation",
                TOOLS,
            )

    assert asyncio.run(run()) == ToolCommand(tool="finish")
    body = seen[0]
    assert body["messages"][0] == {"role": "system", "content": "Owned instructions"}
    assert body["messages"][-1] == {
        "role": "tool",
        "content": "Result",
        "tool_call_id": "call_0",
    }
    assert body["messages"][-2]["tool_calls"][0]["function"]["arguments"] == {
        "file": "/tmp/a"
    }
    assert body["tools"] == [convert_to_openai_tool(tool) for tool in TOOLS]
    assert not body["messages"][0]["content"].startswith(TOOLS_HEADER)


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


@pytest.mark.parametrize("value", ["42", "null", "true"])
def test_string_parameters_stay_strings_even_when_they_look_like_json(value):
    reply = call("shell_exec_command", command=value)
    assert invoke(completed(reply)).arguments == {"command": value}


@pytest.mark.parametrize("image_job_id", [None, "10000000-0000-4000-8000-000000000001"])
def test_nullable_image_id_retains_null_or_uuid(image_job_id):
    reply = call(
        "send_message", content="Hello", end_run=True, image_job_id=image_job_id
    )
    command = invoke(completed(reply))
    assert command.arguments["image_job_id"] == image_job_id
    tool = next(tool for tool in CONVERSATION_TOOLS if tool.name == "send_message")
    validated = tool.args_schema.model_validate(command.arguments)
    assert validated.model_dump(mode="json")["image_job_id"] == image_job_id


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


@pytest.mark.parametrize(
    ("key", "first", "second"),
    [
        ("file", "/tmp/first", "/tmp/second"),
        ("file", "/tmp/first", "/tmp/first"),
        ("file", "", "/tmp/second"),
        ("sudo", "false", "true"),
        ("start_line", "null", "1"),
    ],
)
def test_duplicate_parameters_are_rejected(key, first, second):
    reply = (
        "<tool_call>\n<function=file_read_file>\n"
        f"<parameter={key}>\n{first}\n</parameter>\n"
        f"<parameter={key}>\n{second}\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    with pytest.raises(ServiceError) as error:
        parse_tool_calls(reply, [convert_to_openai_tool(file_read_file)])
    assert str(error.value) == "The agent returned an invalid tool command."
    assert error.value.status == 502
    assert error.value.code == "invalid_command"


def test_wrapperless_duplicate_parameters_are_rejected():
    reply = (
        "<function=file_read_file>\n"
        "<parameter=file>\n/tmp/first\n</parameter>\n"
        "<parameter=file>\n/tmp/second\n</parameter>\n"
        "</function>"
    )
    with pytest.raises(ServiceError) as error:
        parse_tool_calls(reply, [convert_to_openai_tool(file_read_file)])
    assert error.value.code == "invalid_command"


@pytest.mark.parametrize("prefix", ["", call("finish") + "\n"])
def test_duplicate_parameters_reject_the_response_before_first_call_selection(prefix):
    reply = (
        prefix + "<tool_call>\n<function=file_read_file>\n"
        "<parameter=file>\n/tmp/first\n</parameter>\n"
        "<parameter=file>\n/tmp/second\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    with pytest.raises(ServiceError) as error:
        invoke(completed(reply))
    assert str(error.value) == "The agent returned an invalid tool command."
    assert error.value.status == 502
    assert error.value.code == "invalid_command"


def test_parameter_names_are_scoped_to_each_call():
    reply = call("file_read_file", file="/tmp/first") + (
        "\n<function=file_read_file>\n"
        "<parameter=file>\n/tmp/second\n</parameter>\n</function>"
    )
    text, calls = parse_tool_calls(reply, [convert_to_openai_tool(file_read_file)])
    assert text == ""
    assert calls == [
        {"name": "file_read_file", "args": {"file": "/tmp/first"}, "id": "call_0"},
        {"name": "file_read_file", "args": {"file": "/tmp/second"}, "id": "call_1"},
    ]
    assert invoke(completed(reply)) == ToolCommand(
        tool="file_read_file", arguments={"file": "/tmp/first"}
    )


def test_structured_parameters_retain_their_json_values():
    arguments = {"path": "/tmp", "pattern": "example", "include": ["*.py", "*.txt"]}
    text, calls = parse_tool_calls(
        call("file_grep_files", **arguments), [convert_to_openai_tool(file_grep_files)]
    )
    assert text == ""
    assert calls[0]["args"] == arguments


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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("location", ["direct", "object", "array"])
def test_nonfinite_command_arguments_are_rejected_before_checkpoint(value, location):
    argument = {"direct": value, "object": {"value": value}, "array": [value]}[location]
    arguments = {"command": "inert", "timeout": argument}
    with pytest.raises(ValidationError):
        ToolCommand(tool="bash_exec", arguments=arguments)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("location", ["direct", "object", "array"])
def test_saved_nonfinite_command_maps_to_invalid_command(value, location):
    argument = {"direct": value, "object": {"value": value}, "array": [value]}[location]
    arguments = {"command": "inert", "timeout": argument}
    with pytest.raises(ServiceError) as error:
        parse_command(json.dumps({"tool": "bash_exec", "arguments": arguments}))
    assert str(error.value) == "The agent returned an invalid tool command."
    assert error.value.status == 502
    assert error.value.code == "invalid_command"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_streamed_command_is_an_invalid_command(value):
    reply = call("shell_exec_command", command="inert", timeout=value)
    with pytest.raises(ServiceError) as error:
        invoke(completed(reply))
    assert str(error.value) == "The agent returned an invalid tool command."
    assert error.value.status == 502
    assert error.value.code == "invalid_command"


def test_finite_command_arguments_and_nan_text_survive_checkpoint():
    arguments = {
        "command": "NaN",
        "values": [-1.5, 0, "-1.5", "NaN", None, {"value": 2.5}],
    }
    command = ToolCommand(tool="bash_exec", arguments=arguments)
    assert parse_command(command.model_dump_json()).arguments == arguments
    assert invoke(completed(call("shell_exec_command", command="NaN"))).arguments == {
        "command": "NaN"
    }


@pytest.mark.parametrize(
    "ending",
    [
        "",
        chunk(finish="stop"),
        "data: [DONE]\n\n",
        chunk(finish="stop") + "data: [DONE]",
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


@pytest.mark.parametrize("model_id", ["selected-model", "lfm3-1.7b-bf16"])
@pytest.mark.parametrize(
    "reply", [call("finish"), "42", "<tool_call>\n<function=send_message>"]
)
def test_output_limit_is_reported_without_returning_a_command(model_id, reply):
    stream = chunk(reply) + chunk(finish="length") + "data: [DONE]\n\n"
    with pytest.raises(ServiceError) as error:
        invoke(stream, model_id=model_id)
    assert error.value.code == "response_limit"
    assert error.value.status == 502
    assert str(error.value) == (
        "Hikari reached the model's output limit before completing the agent response."
    )


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
