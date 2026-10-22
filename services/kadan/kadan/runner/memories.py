"""Agent-scoped memory access through Tsuchi's existing resource contract."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from ..api.config import ServiceError
from ..api.pydantic_models.resources import Resource, ResourceKind
from ..api.services import parse_resource, require_active

if TYPE_CHECKING:
    from ..api.services import LocalServices


def memory_content(resource: Resource) -> dict:
    return resource.model_dump(
        mode="json", include={"id", "title", "markdown", "tags", "version"}
    )


class AgentMemories:
    """The runner supplies the agent key; tool arguments cannot change it."""

    def __init__(self, services: LocalServices, agent_key: str):
        self.services = services
        self.agent_key = agent_key
        self.path = f"/api/agents/{agent_key}/resources"

    def checked(self, value: object) -> Resource:
        resource = parse_resource(value)
        if resource.agent_key != self.agent_key:
            raise ServiceError("Memory belongs to another agent.", 404, "not_found")
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

    async def get(self, memory_id: UUID) -> Resource:
        value = await self.services.json_request(
            "Tsuchi", "GET", f"{self.path}/{memory_id}"
        )
        return self.checked(value)

    async def read(self, memory_id: UUID) -> dict:
        return memory_content(await self.get(memory_id))

    async def create(self, title: str, markdown: str, tags: list[str]) -> dict:
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
        value = await self.services.json_request(
            "Tsuchi", "PUT", f"{self.path}/{memory_id}", json=content
        )
        return memory_content(self.checked(value))
