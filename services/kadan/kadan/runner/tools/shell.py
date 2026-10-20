"""Shell tools: interactive terminal sessions on the computer."""

from pydantic import Field

from ...api.sandbox_client import SandboxClient, segment
from .base import Arguments, NoArguments, body, computer_tool


class ExecCommandArguments(Arguments):
    command: str = Field(description="Shell command to execute.")
    id: str | None = Field(
        default=None,
        description="Shell session to run in; a new session is created if omitted.",
    )
    exec_dir: str | None = Field(
        default=None, description="Absolute working directory for the command."
    )
    async_mode: bool = Field(
        default=False, description="Return immediately while the command runs."
    )
    timeout: float | None = Field(
        default=None,
        description="Seconds to wait before returning a running status.",
    )
    strict: bool | None = Field(
        default=None,
        description="Fail when the working directory does not exist.",
    )
    no_change_timeout: int | None = Field(
        default=None,
        description="Seconds without new output before returning.",
    )
    hard_timeout: float | None = Field(
        default=None, description="Seconds before the command is stopped."
    )
    preserve_symlinks: bool | None = Field(
        default=False, description="Keep symlinks in the working directory path."
    )
    truncate: bool = Field(
        default=True, description="Truncate output over 30000 characters."
    )


@computer_tool("POST")
async def shell_exec_command(
    args: ExecCommandArguments, computer: SandboxClient
) -> dict:
    """Execute a command in a shell session."""
    return await computer.run("POST", "/v1/shell/exec", json_body=body(args))


class SessionArguments(Arguments):
    id: str = Field(description="Shell session ID.")


@computer_tool("POST")
async def shell_view_shell(args: SessionArguments, computer: SandboxClient) -> dict:
    """View the output of a shell session."""
    return await computer.run("POST", "/v1/shell/view", json_body=body(args))


class WaitArguments(Arguments):
    id: str = Field(description="Shell session ID.")
    seconds: int | None = Field(default=None, description="Seconds to wait.")
    max_wait_seconds: int | None = Field(
        default=None, description="Maximum seconds to wait for completion."
    )


@computer_tool("POST")
async def shell_wait_for_process(args: WaitArguments, computer: SandboxClient) -> dict:
    """Wait for the process in a shell session to return."""
    return await computer.run("POST", "/v1/shell/wait", json_body=body(args))


class WriteArguments(Arguments):
    id: str = Field(description="Shell session ID.")
    input: str = Field(description="Input to write to the process.")
    press_enter: bool = Field(description="Press Enter after the input.")


@computer_tool("POST")
async def shell_write_to_process(args: WriteArguments, computer: SandboxClient) -> dict:
    """Write input to the process in a shell session."""
    return await computer.run("POST", "/v1/shell/write", json_body=body(args))


@computer_tool("POST")
async def shell_kill_process(args: SessionArguments, computer: SandboxClient) -> dict:
    """Terminate the process in a shell session."""
    return await computer.run("POST", "/v1/shell/kill", json_body=body(args))


class CreateSessionArguments(Arguments):
    id: str | None = Field(
        default=None, description="Session ID; generated if omitted."
    )
    exec_dir: str | None = Field(
        default=None, description="Absolute working directory."
    )
    no_change_timeout: int | None = Field(
        default=None,
        description="Seconds without new output before commands return (default 120).",
    )
    preserve_symlinks: bool | None = Field(
        default=False, description="Keep symlinks in the working directory path."
    )


@computer_tool("POST")
async def shell_create_session(
    args: CreateSessionArguments, computer: SandboxClient
) -> dict:
    """Create a shell session, or return the existing one with that ID."""
    return await computer.run("POST", "/v1/shell/sessions/create", json_body=body(args))


class UpdateSessionArguments(Arguments):
    id: str = Field(description="Shell session ID.")
    no_change_timeout: int | None = Field(
        default=None, description="Seconds without new output before returning."
    )


@computer_tool("POST")
async def shell_update_session(
    args: UpdateSessionArguments, computer: SandboxClient
) -> dict:
    """Update a shell session's configuration."""
    return await computer.run("POST", "/v1/shell/sessions/update", json_body=body(args))


@computer_tool("GET")
async def shell_list_sessions(args: NoArguments, computer: SandboxClient) -> dict:
    """List all active shell sessions."""
    return await computer.run("GET", "/v1/shell/sessions")


@computer_tool("GET")
async def shell_get_session_stats(args: NoArguments, computer: SandboxClient) -> dict:
    """Return aggregate statistics for shell sessions."""
    return await computer.run("GET", "/v1/shell/sessions/stats")


class CleanupSessionArguments(Arguments):
    session_id: str = Field(description="Shell session ID.")


@computer_tool("DELETE")
async def shell_cleanup_session(
    args: CleanupSessionArguments, computer: SandboxClient
) -> dict:
    """Close one shell session."""
    return await computer.run(
        "DELETE", f"/v1/shell/sessions/{segment(args.session_id)}"
    )


@computer_tool("DELETE")
async def shell_cleanup_all_sessions(
    args: NoArguments, computer: SandboxClient
) -> dict:
    """Close every shell session."""
    return await computer.run("DELETE", "/v1/shell/sessions")


@computer_tool("GET")
async def shell_get_terminal_url(args: NoArguments, computer: SandboxClient) -> dict:
    """Create a shell session and return its web terminal URL."""
    return await computer.run("GET", "/v1/shell/terminal-url")


TOOLS = [
    # shell_exec_command,
    # shell_view_shell,
    # shell_wait_for_process,
    # shell_write_to_process,
    # shell_kill_process,
    # shell_create_session,
    # shell_update_session,
    # shell_list_sessions,
    # shell_get_session_stats,
    # shell_cleanup_session,
    # shell_cleanup_all_sessions,
    # shell_get_terminal_url,
]
