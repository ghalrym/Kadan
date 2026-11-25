"""Dynamic MCP tool binding, schema validation, replay safety, and result bounding."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid5

from langchain_core.tools import StructuredTool

from ..api.config import ServiceError
from ..api.mcp_presets import T3_WATCH_TRIGGER_TOOLS
from ..api.pydantic_models.mcp import McpServerRecord, McpToolInfo
from .tools.mcp import MCP_GATEWAY_TOOLS

if TYPE_CHECKING:
    from ..api.services import LocalServices

logger = logging.getLogger(__name__)

MAX_MCP_RESULT_CHARS = 24000
MAX_SEARCH_RESULTS = 15


@dataclass(frozen=True)
class McpOrigin:
    agent_key: str
    conversation_id: UUID
    run_id: UUID


def _sanitize_tool_name(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", raw)
    if not cleaned or not re.match(r"^[A-Za-z_]", cleaned):
        cleaned = f"mcp_{cleaned}"
    return cleaned[:64]


def _clean_input_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return {"type": "object", "properties": {}}
    out: dict[str, Any] = {
        "type": "object",
        "properties": schema.get("properties")
        if isinstance(schema.get("properties"), dict)
        else {},
    }
    if isinstance(schema.get("required"), list):
        out["required"] = [r for r in schema["required"] if isinstance(r, str)]
    if "additionalProperties" in schema:
        out["additionalProperties"] = schema["additionalProperties"]
    return out


def build_mcp_tools(
    servers: list[McpServerRecord], reserved_names: set[str]
) -> tuple[list[StructuredTool], dict[str, tuple[McpServerRecord, McpToolInfo]]]:
    """Build direct-bound MCP StructuredTools + optional 2-tool discovery gateway."""
    bound: list[StructuredTool] = []
    registry: dict[str, tuple[McpServerRecord, McpToolInfo]] = {}
    taken = set(reserved_names)
    has_unpinned = False

    for server in servers:
        if not server.enabled or server.status != "connected":
            continue
        pinned_set = set(server.pinned_tools)
        tools_by_name = {tool.name: tool for tool in server.tools}
        if len(tools_by_name) > len(pinned_set & set(tools_by_name)):
            has_unpinned = True
        for tool_name in server.pinned_tools:
            info = tools_by_name.get(tool_name)
            if info is None:
                continue
            if server.preset == "t3_code" and tool_name not in taken:
                binding_name = _sanitize_tool_name(tool_name)
            else:
                binding_name = _sanitize_tool_name(
                    f"mcp__{server.slug.replace('-', '_')}__{tool_name}"
                )
            if binding_name in taken:
                continue
            taken.add(binding_name)
            schema = _clean_input_schema(info.input_schema)
            desc = (
                info.description.strip()
                or f"Call '{info.name}' on connected MCP server '{server.name}'."
            )

            async def _noop(**kwargs: Any) -> dict:
                return kwargs

            bound.append(
                StructuredTool.from_function(
                    coroutine=_noop,
                    name=binding_name,
                    description=desc,
                    args_schema=schema,
                    metadata={"mcp_server_id": str(server.id), "mcp_tool": info.name},
                )
            )
            registry[binding_name] = (server, info)

    if has_unpinned:
        for gateway_tool in MCP_GATEWAY_TOOLS:
            if gateway_tool.name not in taken:
                taken.add(gateway_tool.name)
                bound.append(gateway_tool)

    return bound, registry


def validate_json_schema_arguments(
    schema: dict[str, Any], arguments: dict[str, Any]
) -> list[tuple[str, str]]:
    """Lightweight JSON-schema check returning (field, code) pairs on violations."""
    if not isinstance(arguments, dict):
        return [("arguments", "dict_type")]
    problems: list[tuple[str, str]] = []
    required = schema.get("required")
    if isinstance(required, list):
        for key in required:
            if isinstance(key, str) and key not in arguments:
                problems.append((key, "missing"))
    properties = schema.get("properties")
    if isinstance(properties, dict):
        if schema.get("additionalProperties") is False:
            for key in arguments:
                if key not in properties:
                    problems.append((str(key), "extra_forbidden"))
        for key, value in arguments.items():
            prop = properties.get(key)
            if not isinstance(prop, dict) or value is None:
                continue
            expected = prop.get("type")
            if expected == "string" and not isinstance(value, str):
                problems.append((str(key), "string_type"))
            elif expected == "integer" and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                problems.append((str(key), "int_type"))
            elif expected == "number" and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                problems.append((str(key), "float_type"))
            elif expected == "boolean" and not isinstance(value, bool):
                problems.append((str(key), "bool_type"))
            elif expected == "array" and not isinstance(value, list):
                problems.append((str(key), "list_type"))
            elif expected == "object" and not isinstance(value, dict):
                problems.append((str(key), "dict_type"))
            enum_values = prop.get("enum")
            if isinstance(enum_values, list) and value not in enum_values:
                problems.append((str(key), "literal_error"))
    return problems[:5]


def search_mcp_tools(
    servers: list[McpServerRecord],
    *,
    server_filter: str | None = None,
    query: str = "",
) -> dict[str, Any]:
    needle = query.strip().lower()
    server_needle = server_filter.strip().lower() if server_filter else ""
    matches: list[dict[str, Any]] = []
    available_servers: list[dict[str, Any]] = []

    for server in servers:
        if not server.enabled or server.status != "connected":
            continue
        available_servers.append(
            {"slug": server.slug, "name": server.name, "tool_count": len(server.tools)}
        )
        if server_needle and server_needle not in {
            server.slug.lower(),
            server.name.lower(),
        }:
            continue
        for tool in server.tools:
            haystack = f"{tool.name} {tool.description}".lower()
            if needle and needle not in haystack:
                continue
            matches.append(
                {
                    "server": server.slug,
                    "tool": tool.name,
                    "description": tool.description[:300],
                    "parameters": _clean_input_schema(tool.input_schema),
                    "read_only": tool.read_only,
                }
            )
            if len(matches) >= MAX_SEARCH_RESULTS:
                break
        if len(matches) >= MAX_SEARCH_RESULTS:
            break

    return {
        "servers": available_servers,
        "matches": matches,
        "truncated": len(matches) >= MAX_SEARCH_RESULTS,
    }


def find_server_tool(
    servers: list[McpServerRecord], server_ref: str, tool_name: str
) -> tuple[McpServerRecord, McpToolInfo] | None:
    needle = server_ref.strip().lower()
    for server in servers:
        if not server.enabled or server.status != "connected":
            continue
        if needle in {server.slug.lower(), server.name.lower()}:
            for tool in server.tools:
                if tool.name == tool_name:
                    return server, tool
    return None


def _truncate_result(result: dict[str, Any]) -> dict[str, Any]:
    serialized = json.dumps(result, ensure_ascii=False, default=str)
    if len(serialized) <= MAX_MCP_RESULT_CHARS:
        return result
    return {
        "truncated": True,
        "excerpt": serialized[: MAX_MCP_RESULT_CHARS - 200],
        "note": f"MCP response exceeded {MAX_MCP_RESULT_CHARS} characters and was truncated.",
    }


async def _maybe_register_t3_watch(
    services: LocalServices,
    server: McpServerRecord,
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
    action_id: UUID,
    origin: McpOrigin | None,
) -> bool:
    if (
        origin is None
        or server.preset != "t3_code"
        or tool_name not in T3_WATCH_TRIGGER_TOOLS
        or "error" in result
    ):
        return False
    # Idle thread creation without an initial message has no active run to watch.
    if tool_name == "t3_thread_launch" and not arguments.get("message"):
        return False
    thread_id = (
        result.get("threadId")
        or result.get("thread_id")
        or arguments.get("threadId")
    )
    if not isinstance(thread_id, str) or not thread_id.strip():
        return False
    run_id = result.get("runId") or result.get("run_id")
    env_id = result.get("environmentId") or arguments.get("environmentId")
    link = result.get("link")
    title = result.get("title") or arguments.get("title")
    watch_id = uuid5(action_id, "t3-watch")
    try:
        await services.json_request(
            "Tsuchi",
            "POST",
            "/api/t3-watches",
            json={
                "id": str(watch_id),
                "server_id": str(server.id),
                "thread_id": thread_id.strip(),
                "run_id": run_id if isinstance(run_id, str) and run_id else None,
                "environment_id": env_id if isinstance(env_id, str) and env_id else None,
                "thread_link": link if isinstance(link, str) and link else None,
                "thread_title": title if isinstance(title, str) and title else None,
                "origin_agent_key": origin.agent_key,
                "origin_conversation_id": str(origin.conversation_id),
                "origin_run_id": str(origin.run_id),
            },
        )
        return True
    except ServiceError as error:
        logger.warning("Could not register T3 thread watch for %s: %s", thread_id, error)
        return False


async def execute_mcp_tool(
    services: LocalServices,
    server: McpServerRecord,
    tool_info: McpToolInfo,
    arguments: dict[str, Any],
    *,
    action_id: UUID,
    replayed: bool,
    origin: McpOrigin | None = None,
) -> dict[str, Any]:
    call_args = dict(arguments)
    injected_idempotency = False
    if tool_info.accepts_client_request_id and not call_args.get("clientRequestId"):
        call_args["clientRequestId"] = str(action_id)
        injected_idempotency = True

    replay_safe = tool_info.read_only or tool_info.idempotent or injected_idempotency
    if replayed and not replay_safe:
        return {
            "error": f"The runner restarted before '{tool_info.name}' on {server.name} "
            "reported a result. It may or may not have run; check before retrying."
        }

    try:
        result = await services.mcp.call_tool(server, tool_info.name, call_args)
    except ServiceError as error:
        return {"error": str(error)}

    watching = await _maybe_register_t3_watch(
        services, server, tool_info.name, call_args, result, action_id, origin
    )
    if watching and isinstance(result, dict):
        result = {
            **result,
            "background_watch": True,
            "note": "Kadan is watching this T3 Code thread in the background and will wake "
            "you in this conversation when it finishes or asks a question. Tell the user "
            "(include the thread link verbatim) and end this turn without polling.",
        }
    return _truncate_result(result)
