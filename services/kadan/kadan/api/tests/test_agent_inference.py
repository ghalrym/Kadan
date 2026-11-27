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
    REASONING_TOKENS,
    REPLY_TOKENS,
    AgentInference,
    HikariChatModel,
    ToolCommand,
)
from kadan.runner.tools import CONVERSATION_TOOLS

TOOLS = CONVERSATION_TOOLS
FILE_ID = "00000000-0000-4000-8000-000000000001"


def chunk(content="", finish=None):
    delta = (
        {"tool_calls": content} if isinstance(content, list) else {"content": content}
    )
    return (
        "data: "
        + json.dumps({"choices": [{"delta": delta, "finish_reason": finish}]})
        + "\n\n"
    )


def completed(content):
    finish = "tool_calls" if isinstance(content, list) else "stop"
    return chunk(content) + chunk(finish=finish) + "data: [DONE]\n\n"


def call(name, **arguments):
    return [
        {
            "id": "call_0",
            "type": "function",
            "index": 0,
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
    ]


def invoke(
    stream,
    status=200,
    media_type="text/event-stream",
    seen=None,
    model_id="inkling-2-mini-iq4-nl",
    bound_tools=TOOLS,
    history=None,
):
    async def run():
        def respond(request):
            body = json.loads(request.content)
            if seen is not None:
                seen.append(body)
            if model_id is None:
                assert "model" not in body
            else:
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


def test_plain_answer_after_tool_uses_existing_reply_action():
    history = [
        SystemMessage(content="Instructions"),
        HumanMessage(content="What is the report called?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_workspace_file",
                    "args": {"file_id": FILE_ID},
                    "id": "call_0",
                }
            ],
        ),
        ToolMessage(content="Google search report", tool_call_id="call_0"),
    ]
    answer = 'One clearly visible word is **"Google"**.'
    assert invoke(completed(answer), history=history).command == ToolCommand(
        tool="send_message",
        arguments={"content": answer, "end_run": True},
    )


@pytest.mark.parametrize(
    "model_id", ["inkling-2-mini-iq4-nl", "inkling-2-mini-q8-0", None]
)
def test_requests_cap_reasoning_and_reply_tokens(model_id):
    seen = []
    invoke(completed("Hello"), model_id=model_id, seen=seen)
    assert seen[0]["max_tokens"] == REASONING_TOKENS + REPLY_TOKENS
    assert seen[0]["reasoning_tokens"] == REASONING_TOKENS
    assert REPLY_TOKENS * 3 >= MAX_COMMAND_CHARACTERS


@pytest.mark.parametrize("answer", ["", "   ", call("finish")])
def test_empty_answer_and_explicit_finish_remain_silent(answer):
    assert invoke(completed(answer)).command == ToolCommand(tool="finish")


def test_plain_answer_cannot_create_unbound_reply_action():
    assert invoke(
        completed("A plain answer"),
        bound_tools=[tool for tool in TOOLS if tool.name == "read_task_status"],
    ).command == ToolCommand(tool="finish")


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
def test_literal_tool_markup_is_preserved_as_plain_answer(answer):
    assert invoke(completed(answer)).command.arguments["content"] == answer


def test_sends_structured_tools_and_true_tool_result_roles():
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
                                "name": "read_workspace_file",
                                "args": {"file_id": FILE_ID},
                                "id": "call_0",
                            }
                        ],
                    ),
                    ToolMessage(content="Result", tool_call_id="call_0"),
                ],
                "inkling-2-mini-iq4-nl",
                "test-conversation",
                TOOLS,
            )

    assert asyncio.run(run()).command == ToolCommand(tool="finish")
    body = seen[0]
    assert body["messages"][0] == {"role": "system", "content": "Owned instructions"}
    assert body["messages"][-1] == {
        "role": "tool",
        "content": "Result",
        "tool_call_id": "call_0",
    }
    assert body["messages"][-2]["tool_calls"][0]["function"]["arguments"] == {
        "file_id": FILE_ID
    }
    assert body["tools"] == [convert_to_openai_tool(tool) for tool in TOOLS]


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
            call("read_task_status", offset=30),
            ToolCommand(tool="read_task_status", arguments={"offset": 30}),
        ),
    ],
)
def test_tool_calls_are_parsed_with_schema_typed_arguments(reply, expected):
    assert invoke(completed(reply)).command == expected


@pytest.mark.parametrize("value", ["42", "null", "true"])
def test_string_parameters_stay_strings_even_when_they_look_like_json(value):
    reply = call("create_workspace_document", title=value)
    assert invoke(completed(reply)).command.arguments == {"title": value}


@pytest.mark.parametrize("image_job_id", [None, "10000000-0000-4000-8000-000000000001"])
def test_nullable_image_id_retains_null_or_uuid(image_job_id):
    reply = call(
        "send_message", content="Hello", end_run=True, image_job_id=image_job_id
    )
    command = invoke(completed(reply)).command
    assert command.arguments["image_job_id"] == image_job_id
    tool = next(tool for tool in CONVERSATION_TOOLS if tool.name == "send_message")
    validated = tool.args_schema.model_validate(command.arguments)
    assert validated.model_dump(mode="json")["image_job_id"] == image_job_id


def test_strings_with_reserved_markup_and_newlines_round_trip():
    content = "  \n<tool_call>\n</parameter>\n<|message_user|>  "
    assert invoke(
        completed(call("send_message", content=content))
    ).command.arguments == {"content": content}
    assert invoke(completed(content)).command.arguments == {
        "content": content,
        "end_run": True,
    }


@pytest.mark.parametrize("value", ['{"file":1,"file":2}', "[]", "{", "null"])
def test_invalid_structured_arguments_reject_the_step(value):
    calls = call("read_workspace_file")
    calls[0]["function"]["arguments"] = value
    with pytest.raises(ServiceError) as error:
        invoke(completed(calls))
    assert error.value.code == "invalid_command"


def test_multiple_actions_are_rejected_before_execution():
    calls = [*call("finish"), *call("send_message", content="x")]
    calls[1].update(id="call_1", index=1)
    with pytest.raises(ServiceError) as error:
        invoke(completed(calls))
    assert error.value.code == "invalid_command"


@pytest.mark.parametrize(
    "reply",
    [
        "Hello there",
        '{"tool":"finish"}',
        "",
    ],
)
def test_replies_without_a_tool_call_finish_the_run_when_unbound(reply):
    assert invoke(
        completed(reply), bound_tools=[tool for tool in TOOLS if tool.name == "read_task_status"]
    ).command == ToolCommand(tool="finish")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("location", ["direct", "object", "array"])
def test_nonfinite_command_arguments_are_rejected_before_checkpoint(value, location):
    argument = {"direct": value, "object": {"value": value}, "array": [value]}[location]
    arguments = {"command": "inert", "timeout": argument}
    with pytest.raises(ValidationError):
        ToolCommand(tool="bash_exec", arguments=arguments)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_streamed_command_is_an_invalid_command(value):
    reply = call("shell_exec_command", command="inert", timeout=value)
    with pytest.raises(ServiceError) as error:
        invoke(completed(reply))
    assert str(error.value) == "The agent returned an invalid tool command."
    assert error.value.status == 502
    assert error.value.code == "invalid_command"


def test_finite_command_arguments_and_nan_text_survive_checkpoint():
    assert invoke(
        completed(call("shell_exec_command", command="NaN"))
    ).command.arguments == {"command": "NaN"}


@pytest.mark.parametrize(
    "ending",
    [
        "",
        chunk(finish="stop"),
        "data: [DONE]\n\n",
        chunk(finish="stop") + "data: [DONE]",
        chunk(finish="bad_reason") + "data: [DONE]\n\n",
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


@pytest.mark.parametrize(
    "model_id", ["inkling-2-mini-iq4-nl", "inkling-2-mini-q8-0", None]
)
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


def test_tool_calls_and_results_replay_as_structured_messages():
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
        ]
    )
    assert wire == [
        {"role": "system", "content": "Instructions"},
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {
                        "name": "shell_exec_command",
                        "arguments": {"command": "ls"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"output": "a.txt"}',
            "tool_call_id": "1",
        },
        {"role": "user", "content": "Thanks"},
    ]


def test_hikari_chat_model_rejects_stop_sequences():
    async def check():
        async with httpx.AsyncClient() as client:
            model = HikariChatModel(
                client=client,
                hikari_url="http://hikari",
                model_id="inkling-2-mini-iq4-nl",
                conversation_key="conversation",
            )
            with pytest.raises(ValueError, match="does not support"):
                await model.ainvoke([HumanMessage(content="Hello")], stop=["end"])

    asyncio.run(check())


def test_hikari_chat_model_preserves_multipart_messages():
    parts = [
        {"type": "text", "text": " before "},
        {"type": "image_url", "image_url": {"url": "image"}},
        {"type": "text", "text": " after "},
    ]
    assert HikariChatModel._wire_messages([HumanMessage(content=parts)]) == [
        {"role": "user", "content": parts}
    ]


def test_priority_and_visible_text_deltas():
    seen = []
    deltas = []

    async def on_delta(text: str):
        deltas.append(text)

    async def check():
        stream = chunk("Hello ") + chunk("world!") + chunk(finish="stop") + "data: [DONE]\n\n"

        def respond(request):
            seen.append(json.loads(request.content))
            return httpx.Response(
                200, text=stream, headers={"content-type": "text/event-stream"}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await AgentInference(client, "http://hikari").call(
                [HumanMessage(content="Hi")],
                "inkling-2-mini-iq4-nl",
                "kadan:test:conversation",
                TOOLS,
                priority="background",
                on_text_delta=on_delta,
            )
            assert result.command == ToolCommand(
                tool="send_message",
                arguments={"content": "Hello world!", "end_run": True},
            )

    asyncio.run(check())
    assert seen[0]["priority"] == "background"
    assert deltas == ["Hello ", "world!"]



def test_hikari_chat_model_rejects_unknown_content_parts():
    with pytest.raises(TypeError):
        HikariChatModel._wire_messages([HumanMessage(content=[{"type": "unknown"}])])


def test_provider_identity_survives_persistence_and_pairs_the_result_after_replay():
    from kadan.api.pydantic_models.agent_runs import AgentRunState, RunResult
    from kadan.runner.transcript import transcript_messages

    reply = call("read_workspace_file", file_id=FILE_ID)
    reply[0]["id"] = "provider-session-412-step-8"
    selected = invoke(completed(reply))
    assert selected.call_id == reply[0]["id"]
    journal = AgentRunState(
        transcript=[
            selected,
            RunResult(call_id=selected.call_id, data={"text": "saved"}),
        ]
    )
    saved = journal.model_dump_json()
    assert "call_id" not in json.loads(saved)["transcript"][0]["command"]
    messages = transcript_messages(AgentRunState.model_validate_json(saved).transcript)
    assert messages[0].tool_calls[0]["id"] == messages[1].tool_call_id == reply[0]["id"]


def test_complete_text_reuse_prefix_false_omits_conversation_id():
    seen = []

    async def check():
        stream = completed("Background result")

        def respond(request):
            seen.append(json.loads(request.content))
            return httpx.Response(
                200, text=stream, headers={"content-type": "text/event-stream"}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            text = await AgentInference(client, "http://hikari").complete_text(
                [HumanMessage(content="Summarize")],
                "inkling-2-mini-iq4-nl",
                conversation_key="kadan:alpha:conv",
                priority="background",
                reuse_prefix=False,
            )
            assert text == "Background result"
            text2 = await AgentInference(client, "http://hikari").complete_text(
                [HumanMessage(content="Summarize again")],
                "inkling-2-mini-iq4-nl",
                conversation_key=None,
                priority="background",
            )
            assert text2 == "Background result"

    asyncio.run(check())
    assert len(seen) == 2
    for payload in seen:
        assert payload["reuse_prefix"] is False
        assert "conversation_id" not in payload
        assert payload["priority"] == "background"


def test_evaluate_decisions_local_and_subscription_fallback():
    seen = []

    async def check():
        def respond(request):
            body = json.loads(request.content)
            seen.append((request.url.path, body))
            if request.url.path == "/v1/decisions":
                return httpx.Response(
                    200,
                    json={
                        "answers": [
                            {"key": "should_save_memory", "type": "Noul", "value": 0.87},
                            {"key": "is_core_memory", "type": "Noul", "value": 0.12},
                        ]
                    },
                )
            return httpx.Response(
                200,
                text=completed('{"should_save_memory": 0.9, "is_core_memory": false}'),
                headers={"content-type": "text/event-stream"},
            )

        questions = [
            {
                "key": "should_save_memory",
                "type": "Noul",
                "instructions": "Did user share a durable fact?",
            },
            {
                "key": "is_core_memory",
                "type": "Noul",
                "instructions": "Is this a core fact?",
            },
        ]
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            inference = AgentInference(client, "http://hikari")
            local_scores = await inference.evaluate_decisions(
                "User: I live in Seattle.",
                questions,
                use_local_decisions=True,
                model_id="inkling-2-mini-iq4-nl",
            )
            fallback_scores = await inference.evaluate_decisions(
                "User: I live in Seattle.",
                questions,
                use_local_decisions=False,
                model_id=None,
            )
            return local_scores, fallback_scores

    local_scores, fallback_scores = asyncio.run(check())
    assert local_scores == {"should_save_memory": 0.87, "is_core_memory": 0.12}
    assert fallback_scores == {"should_save_memory": 0.9, "is_core_memory": 0.0}
    assert seen[0][0] == "/v1/decisions"
    assert seen[1][0] == "/v1/chat/completions"
    assert seen[1][1]["reuse_prefix"] is False
    assert "conversation_id" not in seen[1][1]
    assert seen[1][1]["priority"] == "background"
