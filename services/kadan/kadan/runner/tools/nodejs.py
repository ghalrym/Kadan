"""Node.js tools: JavaScript execution and persistent REPL sessions."""

from pydantic import Field

from ...api.sandbox_client import SandboxClient, segment
from .base import Arguments, NoArguments, body, computer_tool

VERSION = 'Node.js version: "node20", "node22", "node24", or "20", "22", "24".'


class ExecuteArguments(Arguments):
    code: str = Field(description="JavaScript code to execute.")
    timeout: int | None = Field(default=30, description="Timeout in seconds.")
    stdin: str | None = Field(default=None, description="Standard input.")
    files: dict[str, str] | None = Field(
        default=None, description="Extra files to create, by name and content."
    )
    stateful: bool | None = Field(
        default=False, description="Use a persistent REPL session."
    )
    session_id: str | None = Field(
        default=None, description="REPL session for stateful execution."
    )
    cwd: str | None = Field(default=None, description="Working directory.")
    version: str | None = Field(default=None, description=VERSION)


@computer_tool("POST")
async def nodejs_execute_nodejs_code(
    args: ExecuteArguments, computer: SandboxClient
) -> dict:
    """Execute JavaScript with Node.js, fresh each time unless stateful."""
    return await computer.run("POST", "/v1/nodejs/execute", json_body=body(args))


@computer_tool("GET")
async def nodejs_nodejs_info(args: NoArguments, computer: SandboxClient) -> dict:
    """Get Node.js and npm versions and the installed packages."""
    return await computer.run("GET", "/v1/nodejs/info")


class VersionArguments(Arguments):
    version: str | None = Field(default=None, description=VERSION)


class CreateSessionArguments(VersionArguments):
    session_id: str | None = Field(
        default=None, description="Session ID; generated if omitted."
    )
    cwd: str | None = Field(default=None, description="Working directory.")
    max_idle_time: int | None = Field(
        default=86400, description="Seconds idle before the session ends."
    )


@computer_tool("POST")
async def nodejs_create_nodejs_session(
    args: CreateSessionArguments, computer: SandboxClient
) -> dict:
    """Create a persistent Node.js REPL session."""
    return await computer.run(
        "POST",
        "/v1/nodejs/sessions",
        params=body(VersionArguments(version=args.version)),
        json_body=args.model_dump(mode="json", exclude={"version"}, exclude_none=True),
    )


@computer_tool("GET")
async def nodejs_list_nodejs_sessions(
    args: VersionArguments, computer: SandboxClient
) -> dict:
    """List active Node.js REPL sessions."""
    return await computer.run("GET", "/v1/nodejs/sessions", params=body(args))


class SessionArguments(VersionArguments):
    session_id: str = Field(description="REPL session ID.")


@computer_tool("GET")
async def nodejs_get_nodejs_session(
    args: SessionArguments, computer: SandboxClient
) -> dict:
    """Get information about a Node.js REPL session."""
    return await computer.run(
        "GET",
        f"/v1/nodejs/sessions/{segment(args.session_id)}",
        params=body(VersionArguments(version=args.version)),
    )


class UpdateSessionArguments(SessionArguments):
    max_idle_time: int | None = Field(
        default=None, description="New seconds idle before the session ends."
    )
    cwd: str | None = Field(default=None, description="New working directory.")


@computer_tool("PATCH")
async def nodejs_update_nodejs_session(
    args: UpdateSessionArguments, computer: SandboxClient
) -> dict:
    """Update a Node.js REPL session's idle time or working directory."""
    return await computer.run(
        "PATCH",
        f"/v1/nodejs/sessions/{segment(args.session_id)}",
        params=body(VersionArguments(version=args.version)),
        json_body=args.model_dump(
            mode="json", exclude={"session_id", "version"}, exclude_none=True
        ),
    )


@computer_tool("DELETE")
async def nodejs_delete_nodejs_session(
    args: SessionArguments, computer: SandboxClient
) -> dict:
    """End a Node.js REPL session."""
    return await computer.run(
        "DELETE",
        f"/v1/nodejs/sessions/{segment(args.session_id)}",
        params=body(VersionArguments(version=args.version)),
    )


TOOLS = [
    # nodejs_execute_nodejs_code,
    # nodejs_nodejs_info,
    # nodejs_create_nodejs_session,
    # nodejs_list_nodejs_sessions,
    # nodejs_get_nodejs_session,
    # nodejs_update_nodejs_session,
    # nodejs_delete_nodejs_session,
]
