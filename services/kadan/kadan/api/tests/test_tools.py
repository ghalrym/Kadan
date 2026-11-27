import pytest
from pydantic import ValidationError

from kadan.runner.tools import ALL_TOOLS, CONVERSATION_TOOLS, TOOLS_BY_NAME
from kadan.runner.tools.conversation import send_message
from kadan.runner.tools.updates import install_app_update
from kadan.runner.tools.workspace import create_workspace_document


def test_every_tool_has_a_unique_name_and_a_description():
    assert len(ALL_TOOLS) == 27
    assert len(TOOLS_BY_NAME) == len(ALL_TOOLS)
    assert len(CONVERSATION_TOOLS) == 19
    assert "rename_self" in TOOLS_BY_NAME
    assert "check_app_update" in TOOLS_BY_NAME
    assert "install_app_update" in TOOLS_BY_NAME
    for name in (
        "work_project_list",
        "work_thread_list",
        "work_thread_read",
        "work_thread_launch",
        "work_thread_send",
        "work_thread_interrupt",
        "work_pending_request",
        "work_thread_organize",
    ):
        assert name in TOOLS_BY_NAME
    assert "read_request_references" not in TOOLS_BY_NAME
    assert all(tool.description for tool in ALL_TOOLS)


def test_no_computer_tools_are_bound():
    names = {tool.name for tool in ALL_TOOLS}
    assert not names & {
        "use_computer",
        "finish_task",
        "computer_look",
        "save_workspace_file",
        "open_workspace_file",
    }
    assert not any(name.startswith(("browser_", "bash_", "file_")) for name in names)


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (send_message, {"content": "Hi"}),
        (create_workspace_document, {"title": " "}),
        (create_workspace_document, {"title": "x" * 501}),
        (create_workspace_document, {"title": "Notes", "markdown": "x" * 48001}),
        (create_workspace_document, {"title": "Notes", "parent_id": "../alpha"}),
        (create_workspace_document, {"title": "Notes", "agent_key": "other"}),
        (create_workspace_document, {"title": "Notes", "kind": "memory"}),
        (create_workspace_document, {"title": "Notes", "id": "chosen-id"}),
        (install_app_update, {}),
        (install_app_update, {"announcement": "   "}),
        (install_app_update, {"announcement": "Updating", "unknown": True}),
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
    "arguments",
    [
        {},
        {"name": ""},
        {"name": "   "},
        {"name": "x" * 501},
        {"name": "Nova", "agent_key": "other"},
    ],
)
def test_rename_self_tool_rejects_invalid_names_and_scope_override(arguments):
    from kadan.runner.tools.persona import RenameSelfArguments

    with pytest.raises(ValidationError):
        RenameSelfArguments.model_validate(arguments)


def test_rename_self_tool_trims_valid_name():
    from kadan.runner.tools.persona import RenameSelfArguments

    parsed = RenameSelfArguments.model_validate({"name": "  Nova  "})
    assert parsed.name == "Nova"
