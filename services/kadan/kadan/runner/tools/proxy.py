"""Proxy tools: how the computer's traffic is routed."""

from pydantic import Field

from ...api.sandbox_client import SandboxClient, segment
from .base import Arguments, NoArguments, body, computer_tool


@computer_tool("GET")
async def proxy_list_mappings(args: NoArguments, computer: SandboxClient) -> dict:
    """List proxy domain-to-port mappings."""
    return await computer.run("GET", "/v1/proxy/mappings")


class AddMappingArguments(Arguments):
    source: str = Field(description="[protocol://]host[:port][/path]; * is a wildcard.")
    target: str = Field(description="[host:]port[/path]; host defaults to 127.0.0.1.")


@computer_tool("POST")
async def proxy_add_mapping(args: AddMappingArguments, computer: SandboxClient) -> dict:
    """Add or update a proxy mapping, e.g. send *.example.com to a local port."""
    return await computer.run("POST", "/v1/proxy/mappings", json_body=body(args))


class RemoveMappingArguments(Arguments):
    source: str = Field(description="Source pattern of the mapping.")


@computer_tool("DELETE")
async def proxy_remove_mapping(
    args: RemoveMappingArguments, computer: SandboxClient
) -> dict:
    """Remove a proxy mapping by its source pattern."""
    return await computer.run("DELETE", f"/v1/proxy/mappings/{segment(args.source)}")


@computer_tool("GET")
async def proxy_list_excludes(args: NoArguments, computer: SandboxClient) -> dict:
    """List proxy bypass patterns."""
    return await computer.run("GET", "/v1/proxy/excludes")


class ExcludeArguments(Arguments):
    pattern: str = Field(
        description="Domain (*.example.com, .example.com) or CIDR (10.0.0.0/8)."
    )


@computer_tool("POST")
async def proxy_add_exclude(args: ExcludeArguments, computer: SandboxClient) -> dict:
    """Add a bypass pattern; matching hosts connect directly."""
    return await computer.run("POST", "/v1/proxy/excludes", json_body=body(args))


@computer_tool("DELETE")
async def proxy_remove_exclude(args: ExcludeArguments, computer: SandboxClient) -> dict:
    """Remove a bypass pattern."""
    return await computer.run("DELETE", "/v1/proxy/excludes", json_body=body(args))


@computer_tool("GET")
async def proxy_get_upstream(args: NoArguments, computer: SandboxClient) -> dict:
    """Get the upstream proxy configuration."""
    return await computer.run("GET", "/v1/proxy/upstream")


class UpstreamArguments(Arguments):
    server: str = Field(description="host:port or user:pass@host:port.")
    auth_cmd: str | None = Field(
        default=None,
        description='Shell command printing "username:password" for the proxy.',
    )


@computer_tool("PUT")
async def proxy_set_upstream(args: UpstreamArguments, computer: SandboxClient) -> dict:
    """Set the upstream proxy; it takes effect immediately."""
    return await computer.run("PUT", "/v1/proxy/upstream", json_body=body(args))


@computer_tool("DELETE")
async def proxy_remove_upstream(args: NoArguments, computer: SandboxClient) -> dict:
    """Remove the upstream proxy and connect directly."""
    return await computer.run("DELETE", "/v1/proxy/upstream")


class DiagnoseArguments(Arguments):
    url: str = Field(description="URL to check.")


@computer_tool("GET")
async def proxy_diagnose(args: DiagnoseArguments, computer: SandboxClient) -> dict:
    """Show how a URL would be routed through the proxy."""
    return await computer.run("GET", "/v1/proxy/diagnose", params=body(args))


@computer_tool("GET")
async def proxy_health_check(args: NoArguments, computer: SandboxClient) -> dict:
    """Check the proxy subsystem's health."""
    return await computer.run("GET", "/v1/proxy/health")


TOOLS = [
    # proxy_list_mappings,
    # proxy_add_mapping,
    # proxy_remove_mapping,
    # proxy_list_excludes,
    # proxy_add_exclude,
    # proxy_remove_exclude,
    # proxy_get_upstream,
    # proxy_set_upstream,
    # proxy_remove_upstream,
    # proxy_diagnose,
    # proxy_health_check,
]
