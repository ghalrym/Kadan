"""Agent-scoped memory access through Tsuchi's existing resource contract."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING
from uuid import UUID

from ..api.config import ServiceError
from ..api.pydantic_models.resources import Resource, ResourceKind
from ..api.services import parse_resource, require_active

if TYPE_CHECKING:
    from ..api.services import LocalServices


def memory_content(resource: Resource) -> dict:
    fields = {"id", "title", "version", "deleted_at"}
    if resource.deleted_at is None:
        fields.update({"markdown", "tags"})
    return resource.model_dump(mode="json", include=fields)


class AgentMemories:
    """The runner supplies the agent key; tool arguments cannot change it."""

    def __init__(
        self,
        services: LocalServices,
        agent_key: str,
        check_ownership: Callable[[], Awaitable[object]],
    ):
        self.services = services
        self.agent_key = agent_key
        self.check_ownership = check_ownership
        self.path = f"/api/agents/{agent_key}/resources"

    def checked(
        self,
        value: object,
        *,
        memory_id: UUID | None = None,
        include_trashed: bool = False,
    ) -> Resource:
        resource = parse_resource(value)
        if resource.agent_key != self.agent_key:
            raise ServiceError("Memory belongs to another agent.", 404, "not_found")
        if memory_id is not None and resource.id != memory_id:
            raise ServiceError(
                "Tsuchi returned the wrong memory.", 502, "invalid_response"
            )
        if resource.kind != ResourceKind.MEMORY:
            raise ServiceError(
                "This operation requires a memory.", 422, "wrong_resource_kind"
            )
        if not include_trashed:
            require_active(resource, ResourceKind.MEMORY)
        return resource

    async def list(self, query: str, limit: int, offset: int) -> dict:
        rows = await self.services.json_request(
            "Tsuchi",
            "GET",
            self.path,
            params={
                "kind": "memory",
                "q": query,
                "trash": "false",
                "limit": limit,
                "offset": offset,
            },
        )
        if not isinstance(rows, list):
            raise ServiceError(
                "Tsuchi returned an invalid collection.", 502, "invalid_response"
            )
        return {
            "memories": [
                self.checked(row).model_dump(
                    mode="json", include={"id", "title", "tags", "version"}
                )
                for row in rows
            ],
            "offset": offset,
            "limit": limit,
        }

    async def get(self, memory_id: UUID, *, include_trashed: bool = False) -> Resource:
        value = await self.services.json_request(
            "Tsuchi", "GET", f"{self.path}/{memory_id}"
        )
        return self.checked(value, memory_id=memory_id, include_trashed=include_trashed)

    async def read(self, memory_id: UUID, include_trashed: bool = False) -> dict:
        return memory_content(
            await self.get(memory_id, include_trashed=include_trashed)
        )

    async def create(self, title: str, markdown: str, tags: list[str]) -> dict:
        await self.check_ownership()
        value = await self.services.json_request(
            "Tsuchi",
            "POST",
            self.path,
            json={"kind": "memory", "title": title, "markdown": markdown, "tags": tags},
        )
        return memory_content(self.checked(value))

    async def update(
        self,
        memory_id: UUID,
        expected_version: int,
        title: str,
        markdown: str,
        tags: list[str] | None,
    ) -> dict:
        current = await self.get(memory_id)
        require_active(current, ResourceKind.MEMORY, expected_version)
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
        if tags is not None:
            content["tags"] = tags
        await self.check_ownership()
        value = await self.services.json_request(
            "Tsuchi", "PUT", f"{self.path}/{memory_id}", json=content
        )
        return memory_content(self.checked(value, memory_id=memory_id))

    async def forget(self, memory_id: UUID, expected_version: int) -> dict:
        current = await self.get(memory_id)
        require_active(current, ResourceKind.MEMORY, expected_version)
        await self.check_ownership()
        value = await self.services.json_request(
            "Tsuchi",
            "DELETE",
            f"{self.path}/{memory_id}",
            params={"expected_version": expected_version},
        )
        forgotten = self.checked(value, memory_id=memory_id, include_trashed=True)
        if forgotten.deleted_at is None or forgotten.version != expected_version + 1:
            raise ServiceError(
                "Tsuchi did not confirm that the memory was moved to trash. "
                "Read its trash state before retrying.",
                502,
                "invalid_response",
            )
        return memory_content(forgotten)
