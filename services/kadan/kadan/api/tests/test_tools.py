"""The agent's tools, called against a recording sandbox client, never a computer."""

import asyncio

import pytest
from pydantic import ValidationError

from kadan.runner.tools import (
    ALL_TOOLS,
    COMPUTER_TOOLS,
    CONVERSATION_TOOLS,
    TOOLS_BY_NAME,
)
from kadan.runner.tools.browser import (
    browser_click,
    browser_close_tab,
    browser_execute_action,
    browser_get_html,
    browser_navigate,
    browser_wait,
)
from kadan.runner.tools.conversation import send_message
from kadan.runner.tools.file import file_poll_events, file_upload_file
from kadan.runner.tools.mcp import mcp_execute_mcp_tool
from kadan.runner.tools.nodejs import nodejs_update_nodejs_session


class Computer:
    def __init__(self):
        self.calls = []

    async def run(self, method, path, **request):
        self.calls.append((method, path, request))
        return {"status": 200}


def call(tool, /, **arguments):
    computer = Computer()
    asyncio.run(
        tool.ainvoke(arguments, config={"configurable": {"computer": computer}})
    )
    return computer.calls[0]


def required(schema: dict, root: dict) -> dict:
    """The smallest valid arguments for a tool's JSON schema."""
    values = {}
    for name in schema.get("required", []):
        field = schema["properties"][name]
        if "$ref" in field:
            field = root["$defs"][field["$ref"].rsplit("/", 1)[-1]]
        if "enum" in field:
            values[name] = field["enum"][0]
        elif "oneOf" in field or "anyOf" in field:
            values[name] = {"action_type": "WAIT", "duration": 1}
        else:
            values[name] = {
                "string": "x",
                "integer": 1,
                "number": 1.0,
                "boolean": True,
                "array": [],
                "object": {},
            }[field["type"]]
    return values


def test_every_tool_has_a_unique_name_and_a_description():
    assert len(TOOLS_BY_NAME) == len(ALL_TOOLS) == 17
    assert len(CONVERSATION_TOOLS) == 3
    assert all(tool.description for tool in ALL_TOOLS)


@pytest.mark.parametrize("tool", COMPUTER_TOOLS, ids=lambda tool: tool.name)
def test_every_computer_tool_calls_one_route_with_its_declared_method(tool):
    schema = tool.args_schema.model_json_schema()
    method, path, _ = call(tool, **required(schema, schema))
    assert method == tool.metadata["method"]
    assert path.startswith("/v1/")


def test_unset_optional_arguments_are_left_to_the_sandbox():
    method, path, request = call(browser_navigate, url="https://example.com")
    assert (method, path) == ("POST", "/v1/browser/page/navigate")
    assert request["json_body"] == {
        "url": "https://example.com",
        "wait_until": "load",
        "timeout": 30.0,
    }
    _, _, request = call(browser_click, index=4)
    assert request["json_body"] == {"index": 4, "button": "left", "click_count": 1}


def test_path_parameters_are_encoded_and_kept_out_of_the_body():
    _, path, _ = call(browser_close_tab, index=2)
    assert path == "/v1/browser/tabs/2"
    _, path, request = call(file_poll_events, watcher_id="a/b", cursor=5)
    assert path == "/v1/file/watch/a%2Fb/poll"
    assert request["json_body"] == {"cursor": 5, "limit": 100, "timeout": 0}
    _, path, request = call(
        nodejs_update_nodejs_session, session_id="s1", cwd="/tmp", version="22"
    )
    assert path == "/v1/nodejs/sessions/s1"
    assert request["params"] == {"version": "22"}
    assert request["json_body"] == {"cwd": "/tmp"}


def test_query_tools_send_parameters():
    method, path, request = call(browser_get_html, outer=True)
    assert (method, path, request["params"]) == (
        "GET",
        "/v1/browser/page/html",
        {"outer": True},
    )


def test_uploads_send_text_as_a_file():
    _, path, request = call(file_upload_file, file="hello", path="/home/gem/a.txt")
    assert path == "/v1/file/upload"
    assert request["files"] == {"file": ("file", b"hello")}
    assert request["data"] == {"path": "/home/gem/a.txt"}


def test_display_actions_and_mcp_calls_send_their_own_body():
    _, path, request = call(
        browser_execute_action, action={"action_type": "CLICK", "x": 10, "y": 20}
    )
    assert path == "/v1/browser/actions"
    assert request["json_body"] == {
        "action_type": "CLICK",
        "x": 10.0,
        "y": 20.0,
        "button": "left",
        "num_clicks": 1,
    }
    _, path, request = call(
        mcp_execute_mcp_tool,
        server_name="browser",
        tool_name="navigate",
        arguments={"url": "https://example.com"},
    )
    assert path == "/v1/mcp/browser/tools/navigate"
    assert request["json_body"] == {"url": "https://example.com"}


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (browser_navigate, {"address": "https://example.com"}),
        (browser_navigate, {}),
        (browser_wait, {"type": "forever"}),
        (browser_execute_action, {"action": {"action_type": "TELEPORT"}}),
        (send_message, {"content": "Hi"}),
    ],
)
def test_invalid_arguments_are_rejected_by_the_tool_schema(tool, arguments):
    with pytest.raises(ValidationError):
        tool.args_schema.model_validate(arguments)


@pytest.mark.parametrize(
    "changes",
    [
        {"title": "   "},
        {"markdown": "   "},
        {"markdown": "x" * 16001},
        {"expected_version": 0},
        {"expected_version": True},
        {"agent_key": "other"},
    ],
)
def test_persona_tool_rejects_invalid_content_and_scope_override(changes):
    from kadan.runner.tools.persona import UpdatePersonaArguments

    with pytest.raises(ValidationError):
        UpdatePersonaArguments.model_validate(
            {
                "title": "Persona",
                "markdown": "Be concise.",
                "expected_version": None,
                **changes,
            }
        )
