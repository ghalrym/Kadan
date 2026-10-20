"""Sandbox tools: the computer's environment, lifecycle hooks and observability."""

from typing import Literal

from pydantic import Field

from ...api.sandbox_client import SandboxClient, segment
from .base import Arguments, NoArguments, body, computer_tool


@computer_tool("GET")
async def sandbox_get_sandbox_context(
    args: NoArguments, computer: SandboxClient
) -> dict:
    """Get the computer's environment: home directory, versions and so on."""
    return await computer.run("GET", "/v1/sandbox")


@computer_tool("GET")
async def sandbox_python_packages(args: NoArguments, computer: SandboxClient) -> dict:
    """List installed Python packages."""
    return await computer.run("GET", "/v1/sandbox/packages/python")


@computer_tool("GET")
async def sandbox_nodejs_packages(args: NoArguments, computer: SandboxClient) -> dict:
    """List installed Node.js packages."""
    return await computer.run("GET", "/v1/sandbox/packages/nodejs")


# Lifecycle hooks


class RegisterHookArguments(Arguments):
    name: str = Field(description="Unique hook name.")
    command: str = Field(max_length=4096, description="Shell command to run.")
    event: str = Field(default="shutdown", description='Lifecycle event: "shutdown".')
    timeout: float = Field(default=10, description="Seconds the hook may run.")
    priority: int = Field(
        default=100,
        description="Lower runs earlier; equal priorities run in parallel.",
    )


@computer_tool("POST")
async def sandbox_register_hook(
    args: RegisterHookArguments, computer: SandboxClient
) -> dict:
    """Register a command to run on a lifecycle event (currently shutdown)."""
    return await computer.run("POST", "/v1/sandbox/hooks", json_body=body(args))


class ListHooksArguments(Arguments):
    event: str = Field(default="shutdown", description="Lifecycle event.")


@computer_tool("GET")
async def sandbox_list_hooks(args: ListHooksArguments, computer: SandboxClient) -> dict:
    """List registered lifecycle hooks for an event."""
    return await computer.run("GET", "/v1/sandbox/hooks", params=body(args))


class HookArguments(Arguments):
    name: str = Field(description="Hook name.")


@computer_tool("DELETE")
async def sandbox_remove_hook(args: HookArguments, computer: SandboxClient) -> dict:
    """Remove a lifecycle hook. Hooks set by the environment cannot be removed."""
    return await computer.run("DELETE", f"/v1/sandbox/hooks/{segment(args.name)}")


# Observability


class ObserveStartArguments(Arguments):
    mode: Literal["guardrail", "capture"] = Field(
        default="capture", description="Observation mode."
    )
    idempotency_key: str | None = Field(
        default=None, description="Key that makes retries of this start safe."
    )
    duration_seconds: int | None = Field(
        default=None, description="Session length in seconds."
    )
    interval_seconds: float | None = Field(
        default=None, description="Sampling interval in seconds."
    )
    include_processes: bool | None = Field(
        default=None, description="Sample process rows."
    )
    include_disk: bool = Field(default=True, description="Sample disk usage.")


@computer_tool("POST")
async def sandbox_observe_start(
    args: ObserveStartArguments, computer: SandboxClient
) -> dict:
    """Start sampling the computer's resource usage."""
    return await computer.run("POST", "/v1/sandbox/observe/start", json_body=body(args))


class ObserveStopArguments(Arguments):
    session_id: str | None = Field(
        default=None, description="Observation session to stop."
    )


@computer_tool("POST")
async def sandbox_observe_stop(
    args: ObserveStopArguments, computer: SandboxClient
) -> dict:
    """Stop sampling the computer's resource usage."""
    return await computer.run("POST", "/v1/sandbox/observe/stop", json_body=body(args))


@computer_tool("GET")
async def sandbox_observe_status(args: NoArguments, computer: SandboxClient) -> dict:
    """Get the state of resource observation."""
    return await computer.run("GET", "/v1/sandbox/observe/status")


class ObserveLiveArguments(Arguments):
    top_rows: int = Field(default=10, description="Top processes to include.")


@computer_tool("GET")
async def sandbox_observe_live(
    args: ObserveLiveArguments, computer: SandboxClient
) -> dict:
    """Get live resource usage and the top processes."""
    return await computer.run("GET", "/v1/sandbox/observe/live", params=body(args))


class ObserveExportArguments(Arguments):
    idempotency_key: str | None = Field(
        default=None, description="Key that makes retries of this export safe."
    )
    session_id: str | None = Field(
        default=None, description="Observation session to export."
    )
    reason: str = Field(default="manual", description="Why the report is exported.")


@computer_tool("POST")
async def sandbox_observe_export(
    args: ObserveExportArguments, computer: SandboxClient
) -> dict:
    """Export an observation session as a report."""
    return await computer.run(
        "POST", "/v1/sandbox/observe/export", json_body=body(args)
    )


@computer_tool("GET")
async def sandbox_observe_reports(args: NoArguments, computer: SandboxClient) -> dict:
    """List exported observation reports."""
    return await computer.run("GET", "/v1/sandbox/observe/reports")


class ReportArguments(Arguments):
    report_id: str = Field(description="Report ID.")


@computer_tool("GET")
async def sandbox_observe_report_download(
    args: ReportArguments, computer: SandboxClient
) -> dict:
    """Download an observation report."""
    return await computer.run(
        "GET", f"/v1/sandbox/observe/reports/{segment(args.report_id)}"
    )


@computer_tool("DELETE")
async def sandbox_observe_report_delete(
    args: ReportArguments, computer: SandboxClient
) -> dict:
    """Delete an observation report."""
    return await computer.run(
        "DELETE", f"/v1/sandbox/observe/reports/{segment(args.report_id)}"
    )


TOOLS = [
    # sandbox_get_sandbox_context,
    # sandbox_python_packages,
    # sandbox_nodejs_packages,
    # sandbox_register_hook,
    # sandbox_list_hooks,
    # sandbox_remove_hook,
    # sandbox_observe_start,
    # sandbox_observe_stop,
    # sandbox_observe_status,
    # sandbox_observe_live,
    # sandbox_observe_export,
    # sandbox_observe_reports,
    # sandbox_observe_report_download,
    # sandbox_observe_report_delete,
]
