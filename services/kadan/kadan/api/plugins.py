from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import httpx
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, PaginatedRequestParams, TextContent, Tool
from referencing import Registry
from referencing.exceptions import Unresolvable

from .builtin_mcp import create_builtin_server
from .config import ServiceError
from .pydantic_models.plugins import BuiltinServer, PluginManifest, PluginSkill

if TYPE_CHECKING:
    from .services import LocalServices

logger = logging.getLogger(__name__)
DISCOVERY_TIMEOUT_SECONDS = 15
TOOL_TIMEOUT_SECONDS = 30
MAX_PLUGINS = 16
MAX_TOOLS = 64
MAX_DISCOVERY_PAGES = 16
MAX_TOOL_SCHEMA_CHARACTERS = 32000
MAX_TOOL_ARGUMENT_CHARACTERS = 100000
MAX_TOOL_RESULT_CHARACTERS = 32000
TOOL_NAME = re.compile(r"^[a-zA-Z0-9_.-]+$")


def tool_error(message: str) -> CallToolResult:
    return CallToolResult(
        isError=True, content=[TextContent(type="text", text=message)]
    )


def validate_schema(schema: dict[str, Any]) -> Draft202012Validator:
    if len(json.dumps(schema, allow_nan=False)) > MAX_TOOL_SCHEMA_CHARACTERS:
        raise ValueError("Tool schema is too large")
    pending: list[Any] = [schema]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            for key in ("$ref", "$dynamicRef"):
                if key in value and not str(value[key]).startswith("#"):
                    raise ValueError(
                        "Only references within the tool schema are supported"
                    )
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    Draft202012Validator.check_schema(schema)
    # An empty registry never retrieves remote schemas, including during validation.
    return Draft202012Validator(schema, registry=Registry())


@dataclass
class ConnectedTool:
    session: ClientSession
    tool: Tool
    validator: Draft202012Validator


class PluginSession:
    def __init__(self) -> None:
        self.tools: list[Tool] = []
        self.skills: list[PluginSkill] = []
        self._tools: dict[str, ConnectedTool] = {}

    def add_tool(self, plugin_id: str, session: ClientSession, tool: Tool) -> None:
        name = f"{plugin_id}__{tool.name}"
        if not TOOL_NAME.fullmatch(tool.name) or len(tool.name) > 128:
            raise ValueError(
                "MCP tool names must contain at most 128 letters, digits, underscores, dots or hyphens"
            )
        if name in self._tools:
            raise ValueError(f"Duplicate tool {name}")
        if len(self.tools) >= MAX_TOOLS:
            raise ValueError(f"Selected plugins exceed the {MAX_TOOLS}-tool limit")
        if tool.inputSchema.get("type") != "object":
            raise ValueError(f"Tool {name} must accept an object")
        validator = validate_schema(tool.inputSchema)
        if tool.outputSchema is not None:
            validate_schema(tool.outputSchema)
        self._tools[name] = ConnectedTool(session, tool, validator)
        self.tools.append(tool.model_copy(update={"name": name}, deep=True))

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        connected = self._tools.get(name)
        if connected is None:
            return tool_error(f"Tool {name} is not enabled for this chat.")
        try:
            if not isinstance(arguments, dict):
                return tool_error("Tool arguments must be a JSON object.")
            if (
                len(json.dumps(arguments, allow_nan=False))
                > MAX_TOOL_ARGUMENT_CHARACTERS
            ):
                return tool_error("Tool arguments exceed the 100,000-character limit.")
            connected.validator.validate(arguments)
        except ValidationError as error:
            path = ".".join(str(part) for part in error.absolute_path) or "arguments"
            return tool_error(
                f"Invalid tool arguments at {path[:200]}: {error.message[:1000]}"
            )
        except (ValueError, TypeError, RecursionError, Unresolvable):
            return tool_error(
                "Tool arguments or their schema are not valid JSON inputs."
            )
        logger.info("MCP tool started: %s", name)
        try:
            async with asyncio.timeout(TOOL_TIMEOUT_SECONDS):
                result = await connected.session.call_tool(
                    connected.tool.name,
                    arguments,
                    read_timeout_seconds=timedelta(seconds=TOOL_TIMEOUT_SECONDS),
                )
            if len(result.model_dump_json()) > MAX_TOOL_RESULT_CHARACTERS:
                logger.warning("MCP tool result exceeded its size limit: %s", name)
                return tool_error(
                    "The tool result exceeded the 32,000-character limit. Any write may already have completed; read its current state before retrying."
                )
            logger.info("MCP tool finished: %s (error=%s)", name, result.isError)
            return result
        except asyncio.CancelledError:
            logger.info("MCP tool cancelled: %s", name)
            raise
        except Exception as error:
            if isinstance(error, (TimeoutError, httpx.TimeoutException)) or (
                isinstance(error, McpError)
                and error.error.code == httpx.codes.REQUEST_TIMEOUT
            ):
                logger.warning("MCP tool timed out; ending the turn: %s", name)
                # MCP's request timeout only stops waiting. Exit the enclosing
                # session before any further planning or calls can begin.
                raise ServiceError(
                    "The MCP tool timed out and this turn has stopped. The outcome of any write is unknown; read its current state before retrying.",
                    504,
                    "tool_timeout",
                ) from error
            logger.warning("MCP tool failed: %s", name)
            return tool_error(
                "The MCP tool failed. Any write may already have completed; read its current state before retrying."
            )


class PluginRegistry:
    def __init__(self, manifests: list[PluginManifest]):
        if len(manifests) > MAX_PLUGINS:
            raise ValueError(f"Configure at most {MAX_PLUGINS} plugins")
        self._plugins: dict[str, PluginManifest] = {}
        for manifest in manifests:
            if manifest.id in self._plugins:
                raise ValueError(f"Duplicate plugin ID: {manifest.id}")
            self._plugins[manifest.id] = manifest.model_copy(deep=True)

    def catalog(self) -> list[PluginManifest]:
        return [plugin.model_copy(deep=True) for plugin in self._plugins.values()]

    def select(self, plugin_ids: list[str] | None) -> list[str]:
        if plugin_ids is None:
            return [plugin.id for plugin in self._plugins.values() if plugin.enabled]
        if len(plugin_ids) != len(set(plugin_ids)):
            raise ServiceError("Select each plugin only once.", 422, "invalid_plugins")
        for plugin_id in plugin_ids:
            plugin = self._plugins.get(plugin_id)
            if plugin is None or not plugin.enabled:
                raise ServiceError(
                    f"Plugin {plugin_id} is unknown or disabled.",
                    422,
                    "invalid_plugins",
                )
        return list(plugin_ids)

    @asynccontextmanager
    async def connect(
        self, services: LocalServices, key: str, plugin_ids: list[str] | None
    ) -> AsyncIterator[PluginSession]:
        selected = self.select(plugin_ids)
        failure: BaseException | None = None
        try:
            async with AsyncExitStack() as stack:
                try:
                    connected = await self._discover(stack, services, key, selected)
                    yield connected
                except BaseException as error:  # noqa: BLE001 - preserve caller errors after SDK cleanup.
                    # Close SDK task groups normally before propagating caller errors,
                    # so cancellation and ServiceError retain their original type.
                    failure = error
        finally:
            if failure is not None:
                raise failure

    async def _discover(
        self,
        stack: AsyncExitStack,
        services: LocalServices,
        key: str,
        selected: list[str],
    ) -> PluginSession:
        connected = PluginSession()
        for plugin_id in selected:
            plugin = self._plugins[plugin_id]
            try:
                async with asyncio.timeout(DISCOVERY_TIMEOUT_SECONDS):
                    if isinstance(plugin.server, BuiltinServer):
                        session = await stack.enter_async_context(
                            create_connected_server_and_client_session(
                                create_builtin_server(services, key),
                                read_timeout_seconds=timedelta(
                                    seconds=TOOL_TIMEOUT_SECONDS
                                ),
                            )
                        )
                    else:
                        client = await stack.enter_async_context(
                            httpx.AsyncClient(
                                timeout=httpx.Timeout(TOOL_TIMEOUT_SECONDS, connect=3),
                                follow_redirects=False,
                                trust_env=False,
                            )
                        )
                        read, write, _ = await stack.enter_async_context(
                            streamable_http_client(
                                plugin.server.url, http_client=client
                            )
                        )
                        session = await stack.enter_async_context(
                            ClientSession(
                                read,
                                write,
                                read_timeout_seconds=timedelta(
                                    seconds=TOOL_TIMEOUT_SECONDS
                                ),
                            )
                        )
                        await session.initialize()
                    cursor = None
                    seen_cursors: set[str] = set()
                    for _ in range(MAX_DISCOVERY_PAGES):
                        page = await session.list_tools(
                            params=PaginatedRequestParams(cursor=cursor)
                        )
                        for tool in page.tools:
                            connected.add_tool(plugin.id, session, tool)
                        cursor = page.nextCursor
                        if cursor is None:
                            break
                        if cursor in seen_cursors:
                            raise ValueError("MCP tool pagination repeated a cursor")
                        seen_cursors.add(cursor)
                    else:
                        raise ValueError("MCP tool discovery exceeded its page limit")
                    connected.skills.extend(
                        skill.model_copy(deep=True) for skill in plugin.skills
                    )
            except (ValueError, SchemaError) as error:
                raise ServiceError(
                    f"Plugin {plugin.id} has invalid tools: {error}",
                    502,
                    "invalid_plugin_tools",
                ) from error
            except Exception as error:
                raise ServiceError(
                    f"Plugin {plugin.id} could not connect or discover tools. Check its server setup.",
                    503,
                    "plugin_unavailable",
                ) from error
        return connected
