"""An agent's chats and Spaces, stored in Tsuchi as conversations and projects."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
from uuid import UUID

from .config import ServiceError
from .pydantic_models.conversations import ConversationSummary, Space
from .pydantic_models.resources import Resource, ResourceKind

if TYPE_CHECKING:
    from .services import LocalServices

DEFAULT_CONVERSATION_TITLE = "New chat"
# "Chain" titled the single conversation each agent had before multiple chats.
DEFAULT_CONVERSATION_TITLES = frozenset({"Chain", DEFAULT_CONVERSATION_TITLE})
TITLE_LENGTH = 60
PREVIEW_LENGTH = 160
PAGE_SIZE = 200
# Runs append to conversations concurrently, so versioned writes may need retries.
WRITE_ATTEMPTS = 5
RECORD_FIELDS = ("id", "agent_key", "version", "created_at", "updated_at", "deleted_at")


def collapse(text: str) -> str:
    return " ".join(text.split())


def shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def newest_first(resources: Iterable[Resource]) -> list[Resource]:
    by_id = sorted(resources, key=lambda resource: str(resource.id))
    return sorted(by_id, key=lambda resource: resource.updated_at, reverse=True)


def summarize(conversation: Resource, space_ids: set[UUID]) -> ConversationSummary:
    first = next(
        (
            collapse(message.content)
            for message in conversation.messages
            if message.role == "user" and message.content.strip()
        ),
        "",
    )
    title = conversation.title
    if first and title.strip() in DEFAULT_CONVERSATION_TITLES:
        title = shorten(first, TITLE_LENGTH)
    return ConversationSummary(
        id=conversation.id,
        agent_key=conversation.agent_key,
        title=title,
        space_id=conversation.parent_id
        if conversation.parent_id in space_ids
        else None,
        message_count=len(conversation.messages),
        preview=shorten(first, PREVIEW_LENGTH),
        version=conversation.version,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def to_space(project: Resource) -> Space:
    return Space(
        id=project.id,
        agent_key=project.agent_key,
        title=project.title,
        instructions=project.markdown,
        version=project.version,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


def parse(value: object) -> Resource:
    from .services import parse_resource  # services constructs this client

    return parse_resource(value)


def content(raw: dict[str, Any], **changes: Any) -> dict[str, Any]:
    # Edit Tsuchi's own JSON so fields Kadan does not model are kept.
    value = {name: item for name, item in raw.items() if name not in RECORD_FIELDS}
    return {**value, **changes}


CHAT_MISSING = "This chat no longer exists. Reload your chats."
SPACE_MISSING = "This Space no longer exists. Reload your Spaces."


class ConversationClient:
    def __init__(self, services: LocalServices):
        self.services = services

    def path(self, key: str, resource_id: UUID | None = None) -> str:
        base = f"/api/agents/{key}/resources"
        return base if resource_id is None else f"{base}/{resource_id}"

    async def _read(
        self, key: str, resource_id: UUID, kind: ResourceKind, missing: str
    ) -> tuple[Resource, dict[str, Any]]:
        try:
            raw = await self.services.json_request(
                "Tsuchi", "GET", self.path(key, resource_id)
            )
        except ServiceError as error:
            if error.status == 404:
                raise ServiceError(missing, 404, "not_found") from error
            raise
        resource = parse(raw)
        if resource.agent_key != key or resource.id != resource_id:
            raise ServiceError(
                "Tsuchi returned an invalid resource.", 502, "invalid_response"
            )
        if resource.kind != kind or resource.deleted_at is not None:
            raise ServiceError(missing, 404, "not_found")
        return resource, raw

    async def conversation(self, key: str, conversation_id: UUID):
        return await self._read(
            key, conversation_id, ResourceKind.CONVERSATION, CHAT_MISSING
        )

    async def project(self, key: str, space_id: UUID):
        return await self._read(key, space_id, ResourceKind.PROJECT, SPACE_MISSING)

    async def all(self, key: str, kind: ResourceKind) -> list[Resource]:
        found: dict[UUID, Resource] = {}
        offset = 0
        while True:
            page = await self.services.list_resource_kinds(
                key, (kind,), limit=PAGE_SIZE, offset=offset
            )
            for resource in page:
                # Concurrent writes can move a record between pages.
                if (
                    resource.id not in found
                    or resource.version > found[resource.id].version
                ):
                    found[resource.id] = resource
            if len(page) < PAGE_SIZE:
                break
            offset += len(page)
        return newest_first(found.values())

    async def list(self, key: str) -> list[ConversationSummary]:
        conversations, projects = await asyncio.gather(
            self.all(key, ResourceKind.CONVERSATION),
            self.all(key, ResourceKind.PROJECT),
        )
        space_ids = {project.id for project in projects}
        return [summarize(conversation, space_ids) for conversation in conversations]

    async def create(self, key: str, title: str, space_id: UUID | None) -> Resource:
        if space_id is not None:
            await self.project(key, space_id)
        return parse(
            await self.services.json_request(
                "Tsuchi",
                "POST",
                self.path(key),
                json={
                    "kind": ResourceKind.CONVERSATION.value,
                    "title": title,
                    "parent_id": str(space_id) if space_id else None,
                },
            )
        )

    async def update(
        self, key: str, conversation_id: UUID, changes: dict[str, Any]
    ) -> ConversationSummary:
        """Apply only the given title and space_id changes, never losing messages."""
        space_changed = "space_id" in changes
        space_id: UUID | None = changes.get("space_id")
        if space_id is not None:
            await self.project(key, space_id)
        updates: dict[str, Any] = {}
        if "title" in changes:
            updates["title"] = changes["title"]
        if space_changed:
            updates["parent_id"] = str(space_id) if space_id else None
        for attempt in range(WRITE_ATTEMPTS):
            current, raw = await self.conversation(key, conversation_id)
            if not updates:
                updated = current
                break
            try:
                updated = parse(
                    await self.services.json_request(
                        "Tsuchi",
                        "PUT",
                        self.path(key, conversation_id),
                        json=content(raw, **updates, expected_version=current.version),
                    )
                )
                break
            except ServiceError as error:
                if error.status != 409:
                    raise
                if attempt == WRITE_ATTEMPTS - 1:
                    raise ServiceError(
                        "This chat is busy saving a reply. Try again in a moment.",
                        409,
                        "version_conflict",
                    ) from error
        if updated.parent_id is None:
            space_ids: set[UUID] = set()
        elif space_changed:
            space_ids = {updated.parent_id}
        else:
            space_ids = await self._project_ids(key, updated.parent_id)
        return summarize(updated, space_ids)

    async def _project_ids(self, key: str, parent_id: UUID) -> set[UUID]:
        try:
            await self.project(key, parent_id)
        except ServiceError as error:
            if error.status == 404:
                return set()
            raise
        return {parent_id}

    async def trash(self, key: str, conversation_id: UUID) -> None:
        for attempt in range(WRITE_ATTEMPTS):
            current, _ = await self.conversation(key, conversation_id)
            try:
                await self.services.json_request(
                    "Tsuchi",
                    "DELETE",
                    self.path(key, conversation_id),
                    params={"expected_version": current.version},
                )
                return
            except ServiceError as error:
                if error.status != 409:
                    raise
                if attempt == WRITE_ATTEMPTS - 1:
                    raise ServiceError(
                        "This chat is busy saving a reply. Try again in a moment.",
                        409,
                        "version_conflict",
                    ) from error

    async def spaces(self, key: str) -> list[Space]:
        return [
            to_space(project) for project in await self.all(key, ResourceKind.PROJECT)
        ]

    async def space(self, key: str, space_id: UUID) -> Space:
        project, _ = await self.project(key, space_id)
        return to_space(project)

    async def create_space(self, key: str, title: str, instructions: str) -> Space:
        return to_space(
            parse(
                await self.services.json_request(
                    "Tsuchi",
                    "POST",
                    self.path(key),
                    json={
                        "kind": ResourceKind.PROJECT.value,
                        "title": title,
                        "markdown": instructions,
                    },
                )
            )
        )

    async def update_space(
        self,
        key: str,
        space_id: UUID,
        title: str,
        instructions: str,
        expected_version: int,
    ) -> Space:
        _, raw = await self.project(key, space_id)
        # Tsuchi checks the version and reports a conflict in its own words.
        return to_space(
            parse(
                await self.services.json_request(
                    "Tsuchi",
                    "PUT",
                    self.path(key, space_id),
                    json=content(
                        raw,
                        title=title,
                        markdown=instructions,
                        expected_version=expected_version,
                    ),
                )
            )
        )

    async def delete_space(self, key: str, space_id: UUID) -> None:
        """Trash the Space's chats first; Tsuchi refuses to trash a parent of active chats."""
        for attempt in range(WRITE_ATTEMPTS):
            project, _ = await self.project(key, space_id)
            for conversation in await self.all(key, ResourceKind.CONVERSATION):
                if conversation.parent_id != space_id:
                    continue
                try:
                    await self.trash(key, conversation.id)
                except ServiceError as error:
                    if error.status != 404:
                        raise
            try:
                await self.services.json_request(
                    "Tsuchi",
                    "DELETE",
                    self.path(key, space_id),
                    params={"expected_version": project.version},
                )
                return
            except ServiceError as error:
                # A chat was added or the Space changed meanwhile; look again.
                if error.status != 409:
                    raise
                if attempt == WRITE_ATTEMPTS - 1:
                    raise ServiceError(
                        "This Space changed while it was being deleted. Try again.",
                        409,
                        "version_conflict",
                    ) from error
