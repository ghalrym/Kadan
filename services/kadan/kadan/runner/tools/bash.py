"""Bash tools: non-interactive bash sessions with offset-based output."""

from typing import Literal

from pydantic import Field

from ...api.sandbox_client import SandboxClient, segment
from .base import Arguments, NoArguments, body, computer_tool


class ExecArguments(Arguments):
    command: str = Field(description="Shell command to execute.")
    session_id: str | None = Field(
        default=None,
        description="Session to run in; reuse it to continue the same session. "
        "A new session is created if omitted.",
    )
    exec_dir: str | None = Field(
        default=None,
        description="Absolute working directory; also becomes the session default.",
    )
    env: dict[str, str] | None = Field(
        default=None, description="Environment variables for this command only."
    )
    async_mode: bool = Field(
        default=False,
        description="Return immediately; read results with bash_output.",
    )
    timeout: float | None = Field(
        default=None,
        description="Seconds to wait before returning a running status.",
    )
    hard_timeout: float | None = Field(
        default=None, description="Seconds before the process is killed."
    )
    max_output_length: int = Field(
        default=50000, description="Maximum characters of stdout and stderr."
    )


@computer_tool("POST")
async def bash_exec(args: ExecArguments, computer: SandboxClient) -> dict:
    """Execute a bash command.

    The result's status is running, completed (check exit_code), timed_out or
    killed. Poll a running command with bash_output.
    """
    return await computer.run("POST", "/v1/bash/exec", json_body=body(args))


class OutputArguments(Arguments):
    session_id: str = Field(description="Bash session ID.")
    command_id: str | None = Field(
        default=None, description="A specific async command; else the session."
    )
    offset: int = Field(default=0, description="Stdout byte offset to read from.")
    stderr_offset: int = Field(default=0, description="Stderr byte offset.")
    wait: bool = Field(
        default=False, description="Long-poll until new output is available."
    )
    wait_timeout: float = Field(
        default=30.0, description="Maximum seconds to wait when wait is true."
    )


@computer_tool("POST")
async def bash_output(args: OutputArguments, computer: SandboxClient) -> dict:
    """Read new output from a bash session, starting at the given offsets."""
    return await computer.run("POST", "/v1/bash/output", json_body=body(args))


class WriteArguments(Arguments):
    session_id: str = Field(description="Bash session ID.")
    input: str = Field(description="Content to write to stdin.")
    command_id: str | None = Field(
        default=None, description="A specific async command; else the current one."
    )


@computer_tool("POST")
async def bash_write(args: WriteArguments, computer: SandboxClient) -> dict:
    """Write input to a bash session's stdin."""
    return await computer.run("POST", "/v1/bash/write", json_body=body(args))


class KillArguments(Arguments):
    session_id: str = Field(description="Bash session ID.")
    signal: Literal["SIGTERM", "SIGKILL", "SIGINT"] = Field(
        default="SIGTERM", description="Signal to send."
    )


@computer_tool("POST")
async def bash_kill(args: KillArguments, computer: SandboxClient) -> dict:
    """Send a signal to a bash session's process."""
    return await computer.run("POST", "/v1/bash/kill", json_body=body(args))


@computer_tool("GET")
async def bash_sessions(args: NoArguments, computer: SandboxClient) -> dict:
    """List bash sessions and whether each is ready or closed."""
    return await computer.run("GET", "/v1/bash/sessions")


class CreateSessionArguments(Arguments):
    session_id: str | None = Field(
        default=None, description="Session ID; generated if omitted."
    )
    exec_dir: str | None = Field(
        default=None, description="Absolute initial working directory."
    )
    snapshot_path: str | None = Field(
        default=None, description="Shell script to source when the session starts."
    )


@computer_tool("POST")
async def bash_create_session(
    args: CreateSessionArguments, computer: SandboxClient
) -> dict:
    """Create a bash session."""
    return await computer.run("POST", "/v1/bash/sessions/create", json_body=body(args))


class CloseSessionArguments(Arguments):
    session_id: str = Field(description="Bash session ID.")


@computer_tool("POST")
async def bash_close_session(
    args: CloseSessionArguments, computer: SandboxClient
) -> dict:
    """Close a bash session."""
    return await computer.run(
        "POST", f"/v1/bash/sessions/{segment(args.session_id)}/close"
    )


TOOLS = [
    bash_exec,
    bash_output,
    bash_write,
    bash_kill,
    # bash_sessions,
    # bash_create_session,
    # bash_close_session,
]
