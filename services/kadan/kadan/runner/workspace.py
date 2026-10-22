"""Project plans and documents use existing, versioned Tsuchi Spaces."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import UUID

from ..api.config import ServiceError
from ..api.pydantic_models.resources import Resource, ResourceKind
from ..api.services import parse_resource, require_active

if TYPE_CHECKING:
    from ..api.services import LocalServices

PROJECT_TAG = "kadan-project"
STATUS_PREFIX = "project-status:"


class ProjectStatus(StrEnum):
    ACTIVE = "active"
    BLOCKED = "blocked"
    AWAITING_DECISION = "awaiting_decision"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


def project_status(resource: Resource) -> ProjectStatus | None:
    if PROJECT_TAG not in resource.tags:
        return None
    states = [
        tag.removeprefix(STATUS_PREFIX)
        for tag in resource.tags
        if tag.startswith(STATUS_PREFIX)
    ]
    if (
        resource.kind != ResourceKind.SPACE
        or len(states) != 1
        or states[0] not in ProjectStatus
    ):
        raise ServiceError(
            "Project status tags changed. Check the plan in Workspace before continuing.",
            409,
        )
    return ProjectStatus(states[0])


def document_content(resource: Resource) -> dict:
    return {
        **resource.model_dump(
            mode="json",
            include={
                "id",
                "kind",
                "title",
                "markdown",
                "parent_id",
                "tags",
                "version",
                "deck",
            },
        ),
        "project_status": project_status(resource),
    }


class AgentWorkspace:
    def __init__(self, services: LocalServices, agent_key: str):
        self.services = services
        self.agent_key = agent_key
        self.path = f"/api/agents/{agent_key}/resources"

    def checked(self, value: object) -> Resource:
        resource = parse_resource(value)
        if resource.agent_key != self.agent_key:
            raise ServiceError("Document belongs to another agent.", 404)
        if resource.kind not in {ResourceKind.SPACE, ResourceKind.DECK}:
            raise ServiceError("This operation requires a Workspace document.", 422)
        require_active(resource, resource.kind)
        return resource

    async def get(self, document_id: UUID) -> Resource:
        return self.checked(
            await self.services.json_request(
                "Tsuchi", "GET", f"{self.path}/{document_id}"
            )
        )

    async def list(
        self,
        *,
        query: str = "",
        limit: int = 20,
        offset: int = 0,
        projects_only: bool = False,
    ) -> dict:
        rows = await self.services.json_request(
            "Tsuchi",
            "GET",
            self.path,
            params={
                "kind": "space",
                "q": f"{PROJECT_TAG} {query}".strip() if projects_only else query,
                "trash": "false",
                "limit": limit,
                "offset": offset,
            },
        )
        if not isinstance(rows, list):
            raise ServiceError("Tsuchi returned an invalid document list.", 502)
        documents = []
        for row in rows:
            resource = self.checked(row)
            status = project_status(resource)
            if projects_only and status is None:
                continue
            documents.append(
                {
                    **resource.model_dump(
                        mode="json", include={"id", "title", "parent_id", "version"}
                    ),
                    "project_status": status,
                }
            )
        return {
            "documents": documents,
            "next_offset": offset + len(rows) if len(rows) == limit else None,
        }

    async def create_project(self, title: str, markdown: str) -> dict:
        return document_content(
            self.checked(
                await self.services.json_request(
                    "Tsuchi",
                    "POST",
                    self.path,
                    json={
                        "kind": "space",
                        "title": title,
                        "markdown": markdown,
                        "tags": [PROJECT_TAG, STATUS_PREFIX + ProjectStatus.ACTIVE],
                    },
                )
            )
        )

    async def update(
        self,
        current: Resource,
        expected_version: int,
        markdown: str,
        status: ProjectStatus | None = None,
    ) -> dict:
        require_active(current, ResourceKind.SPACE, expected_version)
        previous = project_status(current)
        if (previous is None) != (status is None):
            raise ServiceError(
                "Use update_project for project plans and update_workspace_document for other Spaces.",
                422,
            )
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
        if status is not None:
            content["tags"] = [
                tag for tag in current.tags if not tag.startswith(STATUS_PREFIX)
            ] + [STATUS_PREFIX + status]
        content.update(markdown=markdown, expected_version=expected_version)
        return document_content(
            self.checked(
                await self.services.json_request(
                    "Tsuchi", "PUT", f"{self.path}/{current.id}", json=content
                )
            )
        )
