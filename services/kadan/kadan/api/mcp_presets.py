"""Built-in MCP server presets (with T3 Code as the first-class coding preset)."""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

# Curated core T3 Code tools bound directly into the agent prompt by default.
# Keeps prompt schema overhead tiny for local Inkling 2 Mini while still letting
# the agent reach any other T3 Code tool on demand via mcp_search_tools/mcp_call_tool.
T3_CODE_DEFAULT_PINNED_TOOLS: tuple[str, ...] = (
    "t3_project_list",
    "t3_thread_list",
    "t3_thread_search",
    "t3_thread_launch",
    "t3_thread_send",
    "t3_thread_read",
    "t3_thread_interrupt",
    "t3_pending_request",
)

# Optional named groups shown as simple toggles in the Kadan Integrations UI.
T3_CODE_TOOL_GROUPS: dict[str, tuple[str, ...]] = {
    "core": T3_CODE_DEFAULT_PINNED_TOOLS,
    "projects": (
        "t3_project_read",
        "t3_project_create",
        "t3_project_clone",
    ),
    "pull_requests": (
        "pull_request_list",
        "pull_request_read",
        "pull_request_create",
        "ci_logs",
    ),
    "issues": (
        "list_issues",
        "read_issue",
        "create_issue",
    ),
    "schedules": (
        "list_scheduled_tasks",
        "schedule_task",
        "delete_scheduled_task",
    ),
}

# T3 tools whose successful call starts or resumes background work that Kadan watches.
T3_WATCH_TRIGGER_TOOLS: frozenset[str] = frozenset(
    {"t3_thread_launch", "t3_thread_send"}
)


def derive_slug(name: str, preset: str) -> str:
    if preset == "t3_code":
        return "t3-code"
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    slug = slug[:48].strip("-")
    return slug or "mcp-server"


def derive_app_url(mcp_url: str, preset: str) -> str | None:
    """For T3 Code, strip the trailing /mcp path so users get a direct app link."""
    if preset != "t3_code":
        return None
    parsed = urlsplit(mcp_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/mcp"):
        path = path[:-4] or "/"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")


def default_pinned_tools(preset: str, discovered_names: list[str]) -> list[str]:
    """Select default direct-bound tools when a server syncs tools for the first time."""
    if preset == "t3_code":
        available = set(discovered_names)
        return [name for name in T3_CODE_DEFAULT_PINNED_TOOLS if name in available]
    # For small custom MCP servers (<= 8 tools), bind them directly; larger ones use
    # the 2-tool discovery gateway unless the user pins specific tools.
    if len(discovered_names) <= 8:
        return list(discovered_names)
    return []
