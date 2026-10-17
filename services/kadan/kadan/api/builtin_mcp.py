"""Agent-bound MCP access to Tsuchi memories and workspace documents."""

from functools import partial
from typing import Annotated, Any, Protocol
from uuid import UUID

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.fastmcp.tools import Tool
from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase, FuncMetadata
from mcp.types import ToolAnnotations
from pydantic import ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from .config import ServiceError
from .pydantic_models.resources import AgentKey, Resource, ResourceKind

WRITE_OUTCOME_UNKNOWN = (
    "The write may already have completed. "
    "Read or search the current state before retrying."
)


class StorageServices(Protocol):
    async def json_request(
        self, service: str, method: str, path: str, **kwargs: Any
    ) -> Any: ...


class StrictToolMetadata(FuncMetadata):
    def pre_parse_json(self, data: dict[str, Any]) -> dict[str, Any]:
        # MCP already supplies JSON arguments; do not reinterpret string values.
        return data


class ToolInput(ArgModelBase):
    model_config = ConfigDict(extra="forbid", strict=True)


class SearchInput(ToolInput):
    query: str = Field(default="", max_length=500)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=10, ge=1, le=20)


class ReadInput(ToolInput):
    resource_id: UUID = Field(strict=False)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=8000, ge=1, le=12000)


class DocumentInput(ToolInput):
    title: str = Field(min_length=1, max_length=500)
    markdown: str = Field(max_length=48000)

    @field_validator("title")
    @classmethod
    def nonblank_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Title cannot be blank")
        return value.strip()


class CreateInput(DocumentInput):
    tags: list[Annotated[str, Field(min_length=1, max_length=100)]] = Field(
        default_factory=list, max_length=100
    )
    parent_id: Annotated[UUID, Field(strict=False)] | None = None


class UpdateInput(DocumentInput):
    resource_id: UUID = Field(strict=False)
    expected_version: int = Field(ge=1)


def resource_summary(resource: Resource) -> dict[str, Any]:
    return {
        "id": str(resource.id),
        "kind": resource.kind.value,
        "title": resource.title[:500],
        "version": resource.version,
        "updated_at": resource.updated_at.isoformat(),
    }


class BuiltinTools:
    def __init__(self, services: StorageServices, key: str):
        self.services = services
        self.key = TypeAdapter(AgentKey).validate_python(key, strict=True)
        self.path = f"/api/agents/{self.key}/resources"

    async def request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            return await self.services.json_request("Tsuchi", method, path, **kwargs)
        except ServiceError as error:
            message = str(error)
            if method in {"POST", "PUT"} and (
                error.status >= 500 or error.code == "invalid_response"
            ):
                message = f"{message} {WRITE_OUTCOME_UNKNOWN}"
            raise ToolError(message) from error

    def checked_resource(
        self, value: Any, kind: ResourceKind, resource_id: UUID | None = None
    ) -> Resource:
        try:
            resource = Resource.model_validate(value)
        except ValidationError as error:
            raise ToolError("Tsuchi returned an invalid resource.") from error
        if resource.agent_key != self.key:
            raise ToolError(
                "Tsuchi returned a resource outside this agent's namespace."
            )
        if resource.kind != kind:
            raise ToolError(f"This tool requires a {kind.value} resource.")
        if resource_id is not None and resource.id != resource_id:
            raise ToolError("Tsuchi returned a different resource than requested.")
        return resource

    def write_result(
        self, value: Any, kind: ResourceKind, resource_id: UUID | None = None
    ) -> dict[str, Any]:
        try:
            resource = self.checked_resource(value, kind, resource_id)
            self.require_active(resource)
        except ToolError as error:
            raise ToolError(f"{error} {WRITE_OUTCOME_UNKNOWN}") from error
        return resource_summary(resource)

    async def active_resource(self, kind: ResourceKind, resource_id: UUID) -> Resource:
        resource = self.checked_resource(
            await self.request("GET", f"{self.path}/{resource_id}"), kind, resource_id
        )
        self.require_active(resource)
        return resource

    @staticmethod
    def require_active(resource: Resource) -> None:
        if resource.deleted_at is not None:
            raise ToolError("This resource is in trash. Restore it in Tsuchi first.")

    async def search(
        self, kind: ResourceKind, query: str, offset: int, limit: int
    ) -> dict[str, Any]:
        values = await self.request(
            "GET",
            self.path,
            params={
                "kind": kind.value,
                "q": query,
                "trash": "false",
                "offset": offset,
                "limit": limit + 1,
            },
        )
        if not isinstance(values, list) or len(values) > limit + 1:
            raise ToolError("Tsuchi returned an invalid search page.")
        resources = [self.checked_resource(value, kind) for value in values]
        return {
            "items": [
                {
                    **resource_summary(resource),
                    "excerpt": resource.markdown[:300],
                    "total_characters": len(resource.markdown),
                }
                for resource in resources[:limit]
                if resource.deleted_at is None
            ],
            "next_offset": offset + limit if len(resources) > limit else None,
        }

    async def read(
        self, kind: ResourceKind, resource_id: UUID, offset: int, limit: int
    ) -> dict[str, Any]:
        resource = await self.active_resource(kind, resource_id)
        end = min(offset + limit, len(resource.markdown))
        return {
            **resource_summary(resource),
            "markdown": resource.markdown[offset:end],
            "offset": offset,
            "total_characters": len(resource.markdown),
            "next_offset": end if end < len(resource.markdown) else None,
        }

    async def create(
        self,
        kind: ResourceKind,
        title: str,
        markdown: str,
        tags: list[str],
        parent_id: UUID | None,
    ) -> dict[str, Any]:
        if parent_id is not None:
            await self.active_resource(ResourceKind.SPACE, parent_id)
        return self.write_result(
            await self.request(
                "POST",
                self.path,
                json={
                    "kind": kind.value,
                    "title": title,
                    "markdown": markdown,
                    "tags": tags,
                    "parent_id": str(parent_id) if parent_id else None,
                },
            ),
            kind,
        )

    async def update(
        self,
        kind: ResourceKind,
        resource_id: UUID,
        expected_version: int,
        title: str,
        markdown: str,
    ) -> dict[str, Any]:
        current = await self.active_resource(kind, resource_id)
        if current.version != expected_version:
            raise ToolError("This resource changed. Read it again before updating.")
        # Tsuchi updates replace the content object, so retain all unedited metadata.
        content = current.model_dump(
            mode="json",
            exclude={
                "id",
                "agent_key",
                "version",
                "created_at",
                "updated_at",
                "deleted_at",
            },
        )
        content.update(
            title=title, markdown=markdown, expected_version=expected_version
        )
        return self.write_result(
            await self.request("PUT", f"{self.path}/{resource_id}", json=content),
            kind,
            resource_id,
        )


def create_builtin_server(services: StorageServices, key: str) -> FastMCP:
    """Create a server whose tools cannot change their bound agent namespace."""
    operations = BuiltinTools(services, key)
    tools = []
    for prefix, kind, label in (
        ("memory", ResourceKind.MEMORY, "durable memory"),
        ("workspace", ResourceKind.SPACE, "workspace Space document"),
    ):
        for action, model, description in (
            (
                "search",
                SearchInput,
                "Search by text, or list with an empty query. Returns excerpts and next_offset for pagination.",
            ),
            (
                "read",
                ReadInput,
                "Read Markdown by resource UUID. Offset and limit count characters. Continue with next_offset; version identifies the snapshot.",
            ),
            (
                "create",
                CreateInput,
                "Create with title and Markdown (up to 48,000 characters). Optional parent_id must identify an active Space in this agent.",
            ),
            (
                "update",
                UpdateInput,
                "Replace title and full Markdown (up to 48,000 characters). First read the document and supply its expected_version. Retains tags, sources, attachments and parent; conflicts require rereading.",
            ),
        ):
            # Explicit SDK argument models reject extra fields, including attempts to
            # override the bound agent, and prevent coercing invalid primitive types.
            tools.append(
                Tool(
                    name=f"{prefix}_{action}",
                    description=f"{action.capitalize()} this agent's {label}. {description}",
                    fn=partial(getattr(operations, action), kind),
                    parameters=model.model_json_schema(),
                    fn_metadata=StrictToolMetadata(arg_model=model),
                    is_async=True,
                    annotations=ToolAnnotations(
                        readOnlyHint=action in {"search", "read"},
                        destructiveHint=action == "update",
                        openWorldHint=False,
                    ),
                )
            )
    return FastMCP("Kadan memory and workspace", tools=tools)
