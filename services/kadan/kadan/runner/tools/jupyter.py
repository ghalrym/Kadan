"""Jupyter tools: Python kernels whose state persists across calls."""

from pydantic import Field

from ...api.sandbox_client import SandboxClient, segment
from .base import Arguments, NoArguments, body, computer_tool


class ExecuteArguments(Arguments):
    code: str = Field(description="Python code to execute.")
    timeout: int | None = Field(default=30, description="Timeout in seconds.")
    kernel_name: str | None = Field(
        default=None,
        description="python3, python3.10, python3.11 or python3.12; defaults to "
        "the runtime's Python.",
    )
    session_id: str | None = Field(
        default=None, description="Session that keeps kernel state between calls."
    )
    cwd: str | None = Field(default=None, description="Kernel working directory.")


@computer_tool("POST")
async def jupyter_execute_jupyter_code(
    args: ExecuteArguments, computer: SandboxClient
) -> dict:
    """Execute Python code in a Jupyter kernel."""
    return await computer.run("POST", "/v1/jupyter/execute", json_body=body(args))


@computer_tool("GET")
async def jupyter_jupyter_info(args: NoArguments, computer: SandboxClient) -> dict:
    """Get information about available Jupyter kernels."""
    return await computer.run("GET", "/v1/jupyter/info")


class CreateSessionArguments(Arguments):
    session_id: str | None = Field(
        default=None, description="Session ID; generated if omitted."
    )
    kernel_name: str | None = Field(
        default=None, description="python3, python3.10, python3.11 or python3.12."
    )
    cwd: str | None = Field(default=None, description="Working directory.")


@computer_tool("POST")
async def jupyter_create_jupyter_session(
    args: CreateSessionArguments, computer: SandboxClient
) -> dict:
    """Create a Jupyter session."""
    return await computer.run(
        "POST", "/v1/jupyter/sessions/create", json_body=body(args)
    )


@computer_tool("GET")
async def jupyter_list_sessions(args: NoArguments, computer: SandboxClient) -> dict:
    """List active Jupyter sessions."""
    return await computer.run("GET", "/v1/jupyter/sessions")


class SessionArguments(Arguments):
    session_id: str = Field(description="Jupyter session ID.")


@computer_tool("DELETE")
async def jupyter_cleanup_session(
    args: SessionArguments, computer: SandboxClient
) -> dict:
    """Close one Jupyter session."""
    return await computer.run(
        "DELETE", f"/v1/jupyter/sessions/{segment(args.session_id)}"
    )


@computer_tool("DELETE")
async def jupyter_cleanup_all_sessions(
    args: NoArguments, computer: SandboxClient
) -> dict:
    """Close every Jupyter session."""
    return await computer.run("DELETE", "/v1/jupyter/sessions")


TOOLS = [
    # jupyter_execute_jupyter_code,
    # jupyter_jupyter_info,
    # jupyter_create_jupyter_session,
    # jupyter_list_sessions,
    # jupyter_cleanup_session,
    # jupyter_cleanup_all_sessions,
]
