"""The agent's tools, called against a recording sandbox client, never a computer."""

import asyncio
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from kadan.runner.agent_inference import ToolCommand
from kadan.runner.agent_runs import validate
from kadan.runner.tools import (
    ALL_TOOLS,
    COMPUTER_AGENT_TOOLS,
    COMPUTER_TOOLS,
    CONVERSATION_TOOLS,
    TOOLS_BY_NAME,
)
from kadan.runner.tools.base import body
from kadan.runner.tools.bash import bash_exec
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
from kadan.runner.tools.workspace import create_workspace_document


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
    assert len(ALL_TOOLS) == 23
    assert len(TOOLS_BY_NAME) == len(ALL_TOOLS) + len(COMPUTER_AGENT_TOOLS)
    assert len(CONVERSATION_TOOLS) == 12
    computer_names = {tool.name for tool in COMPUTER_TOOLS}
    assert "browser_take_screenshot" not in computer_names
    assert "browser_get_markdown" not in computer_names
    assert {
        "browser_navigate",
        "browser_get_interactive_elements",
        "browser_get_text",
        "browser_evaluate",
        "browser_execute_action",
    } <= computer_names
    assert all(tool.description for tool in [*ALL_TOOLS, *COMPUTER_AGENT_TOOLS])


def test_codex_stays_with_the_conversation_agent():
    names = {tool.name for tool in ALL_TOOLS}
    assert {"codex_start", "codex_threads", "codex_stop", "use_computer"} <= names
    assert not names & {tool.name for tool in COMPUTER_AGENT_TOOLS}


@pytest.mark.parametrize(
    "tool",
    COMPUTER_TOOLS,
    ids=lambda tool: tool.name,
)
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
    schema = browser_execute_action.args_schema.model_json_schema()
    assert "$defs" not in schema
    _, path, request = call(
        browser_execute_action, action_type="CLICK", x=10, y=20
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


def test_browser_get_markdown_drops_ad_urls_and_enforces_total_budget():
    from kadan.runner.tools.browser import (
        PAGE_RESULT_MAX_CHARACTERS,
        browser_get_markdown,
    )

    giant_ad_url = "https://www.googleadservices.com/pagead/aclk?sa=L&ai=" + ("A" * 3000)
    big_markdown = (
        "# YouTube\n[Home](/)\n[Shorts](/shorts)\n"
        + f"[Shop now]({giant_ad_url})\n" * 6
        + ("Navigation boilerplate line.\n" * 2000)
        + "## Results\n[Kadan Demo Video](https://www.youtube.com/watch?v=abc123)\n"
        + "[Another Cool Video](https://www.youtube.com/watch?v=xyz789)\n"
    )
    assert len(big_markdown) > 50000

    class MarkdownComputer:
        async def run(self, method, path, **request):
            assert (method, path) == ("GET", "/v1/browser/page/markdown")
            assert request.get("max_characters") is None
            return {
                "status": 200,
                "content_type": "application/json",
                "body": {"data": {"title": "YouTube", "markdown": big_markdown}},
            }

    links_result = asyncio.run(
        browser_get_markdown.ainvoke(
            {"links_only": True, "query": "watch"},
            config={"configurable": {"computer": MarkdownComputer()}},
        )
    )
    assert links_result["links_count"] == 2
    assert "watch?v=abc123" in links_result["body"]
    assert "googleadservices.com" not in links_result["body"]

    query_result = asyncio.run(
        browser_get_markdown.ainvoke(
            {"query": "Kadan Demo"},
            config={"configurable": {"computer": MarkdownComputer()}},
        )
    )
    assert "Kadan Demo Video" in query_result["body"]
    assert "watch?v=abc123" in query_result["body"]

    default_result = asyncio.run(
        browser_get_markdown.ainvoke(
            {},
            config={"configurable": {"computer": MarkdownComputer()}},
        )
    )
    assert default_result["truncated"] is True
    assert any("watch?v=abc123" in item for item in default_result["links"])
    assert all("googleadservices.com" not in item for item in default_result["links"])
    total_chars = len(default_result["body"]) + sum(
        len(item) for item in default_result["links"]
    )
    assert total_chars <= PAGE_RESULT_MAX_CHARACTERS


def test_browser_get_text_supports_query_and_offset():
    from kadan.runner.tools.browser import (
        PAGE_RESULT_MAX_CHARACTERS,
        browser_get_text,
    )

    lines = [f"Header line {i}" for i in range(400)]
    lines.extend(
        [
            "Show Your Work (Music Video)",
            "Last Echo Ai",
            "3:45 • 12K views • 2 weeks ago",
        ]
    )
    big_text = "\n".join(lines)

    class TextComputer:
        async def run(self, method, path, **request):
            assert (method, path) == ("GET", "/v1/browser/page/text")
            assert request.get("max_characters") is None
            return {
                "status": 200,
                "content_type": "application/json",
                "body": {"data": {"title": "YouTube", "text": big_text}},
            }

    matched = asyncio.run(
        browser_get_text.ainvoke(
            {"query": "Show Your Work"},
            config={"configurable": {"computer": TextComputer()}},
        )
    )
    assert "Show Your Work (Music Video)" in matched["body"]
    assert "Last Echo Ai" in matched["body"]
    assert "3:45" in matched["body"]

    full = asyncio.run(
        browser_get_text.ainvoke(
            {"offset": 100},
            config={"configurable": {"computer": TextComputer()}},
        )
    )
    assert full["offset"] == 100
    assert len(full["body"]) <= PAGE_RESULT_MAX_CHARACTERS


def test_browser_navigate_attaches_compact_page_observation():
    from kadan.api.config import ServiceError

    class NavComputer:
        def __init__(self, fail_markdown=False):
            self.calls = []
            self.fail_markdown = fail_markdown

        async def run(self, method, path, **request):
            self.calls.append((method, path, request))
            if path == "/v1/browser/page/navigate":
                return {"status": 200, "body": {"ok": True}}
            if self.fail_markdown:
                raise ServiceError("markdown unavailable", 503)
            return {
                "status": 200,
                "body": "# Docs\nWelcome!\n[Getting Started Guide](https://example.com/guide)\n",
            }

    comp = NavComputer()
    result = asyncio.run(
        browser_navigate.ainvoke(
            {"url": "https://example.com"},
            config={"configurable": {"computer": comp}},
        )
    )
    assert result["status"] == 200
    assert result["page"]["excerpt"].startswith("# Docs")
    assert result["page"]["links"] == ["[Getting Started Guide](https://example.com/guide)"]

    failing = NavComputer(fail_markdown=True)
    fallback = asyncio.run(
        browser_navigate.ainvoke(
            {"url": "https://example.com"},
            config={"configurable": {"computer": failing}},
        )
    )
    assert fallback == {"status": 200, "body": {"ok": True}}


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (browser_navigate, {"address": "https://example.com"}),
        (browser_navigate, {}),
        (browser_wait, {"type": "forever"}),
        (browser_execute_action, {"action_type": "TELEPORT"}),
        (browser_execute_action, {"action_type": "MOVE_TO", "x": 10}),
        (browser_execute_action, {"action_type": "TYPING"}),
        (browser_execute_action, {"action_type": "WAIT"}),
        (send_message, {"content": "Hi"}),
        (create_workspace_document, {"title": " "}),
        (create_workspace_document, {"title": "x" * 501}),
        (create_workspace_document, {"title": "Notes", "markdown": "x" * 48001}),
        (create_workspace_document, {"title": "Notes", "parent_id": "../alpha"}),
        (create_workspace_document, {"title": "Notes", "agent_key": "other"}),
        (create_workspace_document, {"title": "Notes", "kind": "memory"}),
        (create_workspace_document, {"title": "Notes", "id": "chosen-id"}),
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


@pytest.mark.parametrize(
    "timeout",
    [float("nan"), float("inf"), float("-inf"), "NaN", "Infinity", "-Infinity"],
)
def test_bash_schema_rejects_nonfinite_timeouts(timeout):
    with pytest.raises(ValidationError):
        bash_exec.args_schema.model_validate({"command": "inert", "timeout": timeout})


@pytest.mark.parametrize(
    "timeout",
    [float("nan"), float("inf"), float("-inf"), "NaN", "Infinity", "-Infinity"],
)
def test_bash_validation_returns_nonfinite_timeout_correction_feedback(timeout):
    # Isolate typed tool validation from ToolCommand's JSON boundary.
    command = Mock(
        spec=ToolCommand,
        tool="bash_exec",
        arguments={"command": "inert", "timeout": timeout},
    )
    rejection = validate(command, [bash_exec])
    assert rejection is not None
    assert rejection.tool == "bash_exec"
    assert rejection.problems[0].field.startswith("timeout")
    assert "finite_number" in {problem.code for problem in rejection.problems}
    assert "inert" not in rejection.model_dump_json()


@pytest.mark.parametrize("timeout", [-1.5, 0, 2.5, "-1.5", "0", "2.5", None])
def test_bash_finite_timeouts_and_nan_command_text_remain_valid(timeout):
    command = ToolCommand(
        tool="bash_exec", arguments={"command": "NaN", "timeout": timeout}
    )
    assert validate(command, [bash_exec]) is None
    args = bash_exec.args_schema.model_validate(command.arguments)
    assert args.command == "NaN"
    assert args.timeout == (None if timeout is None else float(timeout))
    assert "timeout" in args.model_fields_set
    assert ("timeout" in body(args)) is (timeout is not None)


def test_bash_omitted_timeout_retains_none_and_is_left_out_of_body():
    command = ToolCommand(tool="bash_exec", arguments={"command": "NaN"})
    assert validate(command, [bash_exec]) is None
    args = bash_exec.args_schema.model_validate(command.arguments)
    assert args.timeout is None
    assert "timeout" not in args.model_fields_set
    assert "timeout" not in body(args)
