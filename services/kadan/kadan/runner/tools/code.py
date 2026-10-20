"""Code tools: run Python or JavaScript through the computer's unified runtime."""

from typing import Literal

from pydantic import Field

from ...api.sandbox_client import SandboxClient
from .base import Arguments, NoArguments, body, computer_tool


class ExecuteArguments(Arguments):
    language: Literal["python", "javascript"] = Field(description="Language.")
    code: str = Field(description="Source code to execute.")
    timeout: int | None = Field(
        default=None, ge=1, le=300, description="Timeout in seconds."
    )
    cwd: str | None = Field(default=None, description="Working directory.")
    stateful: bool = Field(
        default=False,
        description="Keep variables between calls with the same session_id.",
    )
    session_id: str | None = Field(
        default=None, description="Session for stateful execution."
    )


@computer_tool("POST")
async def code_execute_code(args: ExecuteArguments, computer: SandboxClient) -> dict:
    """Run Python or JavaScript code."""
    return await computer.run("POST", "/v1/code/execute", json_body=body(args))


@computer_tool("GET")
async def code_code_info(args: NoArguments, computer: SandboxClient) -> dict:
    """Describe the supported code runtimes and their versions."""
    return await computer.run("GET", "/v1/code/info")


TOOLS = [
    # code_execute_code,
    # code_code_info,
]
