"""An agent's Space library: Tsuchi folders, pages, slide decks and files."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import ValidationError

from .config import ServiceError
from .conversation_client import PAGE_SIZE, WRITE_ATTEMPTS, content, parse
from .pydantic_models.assets import Asset
from .pydantic_models.resources import Resource, ResourceKind
from .pydantic_models.space import SpaceItem, SpacePage, SpaceSlides

if TYPE_CHECKING:
    from .services import LocalServices

ITEM_TYPES = {
    ResourceKind.FOLDER: "folder",
    ResourceKind.SPACE: "page",
    ResourceKind.DECK: "slides",
}
ITEM_KINDS = tuple(ITEM_TYPES)
# Kinds that are not Space items but can still be stored under a page.
OTHER_KINDS = (ResourceKind.MEMORY, ResourceKind.PERSONA, ResourceKind.CONVERSATION)
FOLDERS = frozenset({ResourceKind.FOLDER})
FOLDERS_AND_PAGES = frozenset({ResourceKind.FOLDER, ResourceKind.SPACE})
# Kadan's pet client tags every pet artwork upload with this tag.
PET_TAG = "pet"

ITEM_MISSING = "This item no longer exists."
FOLDER_MISSING = "The folder you chose no longer exists."
FOLDERS_ONLY = "Folders and files can only go inside folders."
INSIDE_ITSELF = "An item can't go inside itself or something it contains."
IN_USE = "This file is attached to a chat or page, so it can't be moved to trash."
BUSY = "This item changed while it was being saved. Try again."
TRASHED_WITH_FOLDER = "This folder changed while it was being deleted. Try again."
RESTORE_FOLDER = "Restore the folder it was in first."
RESTORE_PAGE = "Restore the page it was in first."
RESTORE_FILES = "Restore the files this item uses first."


def is_pet_artwork(asset: Asset) -> bool:
    return PET_TAG in asset.tags


def resource_item(resource: Resource) -> SpaceItem:
    return SpaceItem(
        id=resource.id,
        type=ITEM_TYPES[resource.kind],
        title=resource.title,
        parent_id=resource.parent_id,
        pinned=resource.pinned,
        filename=None,
        media_type=None,
        size_bytes=None,
        version=resource.version,
        created_at=resource.created_at,
        updated_at=resource.updated_at,
        deleted_at=resource.deleted_at,
    )


def asset_item(asset: Asset) -> SpaceItem:
    return SpaceItem(
        id=asset.id,
        type="image" if asset.is_image else "file",
        title=asset.title,
        parent_id=asset.folder_id,
        pinned=asset.pinned,
        filename=asset.filename,
        media_type=asset.media_type,
        size_bytes=asset.size_bytes,
        version=asset.version,
        created_at=asset.created_at,
        updated_at=asset.updated_at,
        deleted_at=asset.deleted_at,
    )


def newest_first(items: Iterable[SpaceItem]) -> list[SpaceItem]:
    by_id = sorted(items, key=lambda item: str(item.id))
    return sorted(by_id, key=lambda item: item.updated_at, reverse=True)


def plain_write_error(error: ServiceError) -> ServiceError:
    """Reword Tsuchi's link rejections for people who never see its storage."""
    detail = str(error)
    if error.status == 400 and "ancestors" in detail:
        return ServiceError(INSIDE_ITSELF, 422, "invalid_parent")
    if error.status == 400 and ("Parent must be" in detail or "Folder must" in detail):
        return ServiceError(FOLDER_MISSING, 404, "not_found")
    if error.status == 404 and "Resource not found" in detail:
        return ServiceError(FOLDER_MISSING, 404, "not_found")
    return error


class Retry(Exception):
    """A concurrent change means the whole operation should look again."""


class SpaceClient:
    def __init__(self, services: LocalServices):
        self.services = services

    def path(self, key: str, *parts: object) -> str:
        return "/".join([f"/api/agents/{key}", *map(str, parts)])

    async def request(self, method: str, path: str, **kwargs: Any) -> Any:
        return await self.services.json_request("Tsuchi", method, path, **kwargs)

    # Reads

    async def _resources(
        self, key: str, trash: bool, kinds: tuple[ResourceKind, ...] = ITEM_KINDS
    ) -> list[Resource]:
        found: dict[UUID, Resource] = {}
        offset = 0
        while True:
            page = await self.request(
                "GET",
                self.path(key, "resources"),
                params={
                    "kinds": [kind.value for kind in kinds],
                    "trash": str(trash).lower(),
                    "limit": PAGE_SIZE,
                    "offset": offset,
                },
            )
            if not isinstance(page, list):
                raise ServiceError(
                    "Tsuchi returned an invalid collection.", 502, "invalid_response"
                )
            for resource in map(parse, page):
                if (
                    resource.agent_key != key
                    or resource.kind not in kinds
                    or (resource.deleted_at is not None) != trash
                ):
                    raise ServiceError(
                        "Tsuchi returned an invalid collection.",
                        502,
                        "invalid_response",
                    )
                # Concurrent writes can move a record between pages.
                if (
                    resource.id not in found
                    or resource.version > found[resource.id].version
                ):
                    found[resource.id] = resource
            if len(page) < PAGE_SIZE:
                return list(found.values())
            offset += len(page)

    async def _assets(self, key: str, trash: bool) -> list[Asset]:
        found: dict[UUID, Asset] = {}
        offset = 0
        while True:
            page = await self.request(
                "GET",
                self.path(key, "assets"),
                params={
                    "trash": str(trash).lower(),
                    "limit": PAGE_SIZE,
                    "offset": offset,
                },
            )
            if not isinstance(page, list):
                raise ServiceError(
                    "Tsuchi returned an invalid collection.", 502, "invalid_response"
                )
            for asset in map(self._parse_asset, page):
                if asset.agent_key != key or (asset.deleted_at is not None) != trash:
                    raise ServiceError(
                        "Tsuchi returned an invalid collection.",
                        502,
                        "invalid_response",
                    )
                if not is_pet_artwork(asset) and (
                    asset.id not in found or asset.version > found[asset.id].version
                ):
                    found[asset.id] = asset
            if len(page) < PAGE_SIZE:
                return list(found.values())
            offset += len(page)

    @staticmethod
    def _parse_asset(value: object) -> Asset:
        try:
            return Asset.model_validate(value)
        except ValidationError as error:
            raise ServiceError(
                "Tsuchi returned an invalid file.", 502, "invalid_response"
            ) from error

    async def _collection(self, key: str, trash: bool) -> list[SpaceItem]:
        resources, assets = await asyncio.gather(
            self._resources(key, trash), self._assets(key, trash)
        )
        return newest_first([*map(resource_item, resources), *map(asset_item, assets)])

    async def items(self, key: str) -> list[SpaceItem]:
        return await self._collection(key, trash=False)

    async def trash(self, key: str) -> list[SpaceItem]:
        return await self._collection(key, trash=True)

    async def _read_resource(
        self, key: str, item_id: UUID
    ) -> tuple[Resource, dict[str, Any]] | None:
        try:
            raw = await self.request("GET", self.path(key, "resources", item_id))
        except ServiceError as error:
            if error.status == 404:
                return None
            raise
        resource = parse(raw)
        if resource.agent_key != key or resource.id != item_id:
            raise ServiceError(
                "Tsuchi returned an invalid resource.", 502, "invalid_response"
            )
        return resource, raw

    async def _read_asset(self, key: str, item_id: UUID) -> Asset | None:
        try:
            raw = await self.request("GET", self.path(key, "assets", item_id))
        except ServiceError as error:
            if error.status == 404:
                return None
            raise
        asset = self._parse_asset(raw)
        if asset.agent_key != key or asset.id != item_id:
            raise ServiceError(
                "Tsuchi returned an invalid file.", 502, "invalid_response"
            )
        return asset

    async def find(
        self, key: str, item_id: UUID, *, trashed: bool | None = False
    ) -> tuple[Resource, dict[str, Any]] | Asset:
        """Resources and assets both use random UUIDs, so one ID names one item."""
        found: tuple[Resource, dict[str, Any]] | Asset | None
        found = await self._read_resource(key, item_id)
        if found is not None:
            resource, _ = found
            deleted = resource.deleted_at is not None
            if resource.kind in ITEM_TYPES and trashed in (None, deleted):
                return found
            raise ServiceError(ITEM_MISSING, 404, "not_found")
        found = await self._read_asset(key, item_id)
        if (
            found is None
            or is_pet_artwork(found)
            or trashed not in (None, found.deleted_at is not None)
        ):
            raise ServiceError(ITEM_MISSING, 404, "not_found")
        return found

    async def _of_kind(
        self, key: str, item_id: UUID, kind: ResourceKind
    ) -> tuple[Resource, dict[str, Any]]:
        found = await self.find(key, item_id)
        if isinstance(found, Asset) or found[0].kind != kind:
            raise ServiceError(ITEM_MISSING, 404, "not_found")
        return found

    async def page(self, key: str, page_id: UUID) -> SpacePage:
        resource, _ = await self._of_kind(key, page_id, ResourceKind.SPACE)
        return SpacePage(
            **resource_item(resource).model_dump(), markdown=resource.markdown
        )

    async def slides(self, key: str, deck_id: UUID) -> SpaceSlides:
        resource, _ = await self._of_kind(key, deck_id, ResourceKind.DECK)
        return SpaceSlides(
            **resource_item(resource).model_dump(),
            slides=resource.deck.slides if resource.deck else [],
        )

    async def _require_parent(
        self, key: str, parent_id: UUID, allowed: frozenset[ResourceKind]
    ) -> None:
        found = await self._read_resource(key, parent_id)
        if found is None:
            raise ServiceError(FOLDER_MISSING, 404, "not_found")
        parent, _ = found
        if parent.deleted_at is not None or parent.kind not in FOLDERS_AND_PAGES:
            raise ServiceError(FOLDER_MISSING, 404, "not_found")
        if parent.kind not in allowed:
            raise ServiceError(FOLDERS_ONLY, 422, "invalid_parent")

    # Writes

    async def _create(self, key: str, body: dict[str, Any]) -> Resource:
        try:
            return parse(
                await self.request("POST", self.path(key, "resources"), json=body)
            )
        except ServiceError as error:
            raise plain_write_error(error) from error

    async def create_folder(
        self, key: str, title: str, parent_id: UUID | None
    ) -> SpaceItem:
        if parent_id is not None:
            await self._require_parent(key, parent_id, FOLDERS)
        folder = await self._create(
            key,
            {
                "kind": ResourceKind.FOLDER.value,
                "title": title,
                "parent_id": str(parent_id) if parent_id else None,
            },
        )
        return resource_item(folder)

    async def create_page(
        self, key: str, title: str, markdown: str, parent_id: UUID | None
    ) -> SpaceItem:
        if parent_id is not None:
            await self._require_parent(key, parent_id, FOLDERS_AND_PAGES)
        page = await self._create(
            key,
            {
                "kind": ResourceKind.SPACE.value,
                "title": title,
                "markdown": markdown,
                "parent_id": str(parent_id) if parent_id else None,
            },
        )
        return resource_item(page)

    async def update_page(
        self, key: str, page_id: UUID, title: str, markdown: str, expected_version: int
    ) -> SpacePage:
        _, raw = await self._of_kind(key, page_id, ResourceKind.SPACE)
        # Tsuchi checks the version and reports a conflict in its own words.
        updated = parse(
            await self.request(
                "PUT",
                self.path(key, "resources", page_id),
                json=content(
                    raw,
                    title=title,
                    markdown=markdown,
                    expected_version=expected_version,
                ),
            )
        )
        return SpacePage(
            **resource_item(updated).model_dump(), markdown=updated.markdown
        )

    async def update(
        self, key: str, item_id: UUID, changes: dict[str, Any]
    ) -> SpaceItem:
        """Change only the given title, parent and pin, retrying lost races."""
        found = await self.find(key, item_id)
        if "parent_id" in changes and changes["parent_id"] is not None:
            is_folder = isinstance(found, Asset) or found[0].kind == ResourceKind.FOLDER
            await self._require_parent(
                key, changes["parent_id"], FOLDERS if is_folder else FOLDERS_AND_PAGES
            )
        for attempt in range(WRITE_ATTEMPTS):
            if attempt:
                found = await self.find(key, item_id)
            try:
                if isinstance(found, Asset):
                    return await self._update_asset(key, found, changes)
                return await self._update_resource(key, *found, changes)
            except ServiceError as error:
                if error.status != 409:
                    raise plain_write_error(error) from error
        raise ServiceError(BUSY, 409, "version_conflict")

    async def _update_resource(
        self, key: str, resource: Resource, raw: dict[str, Any], changes: dict[str, Any]
    ) -> SpaceItem:
        updates = {
            name: changes[name] for name in ("title", "pinned") if name in changes
        }
        if "parent_id" in changes:
            parent_id = changes["parent_id"]
            updates["parent_id"] = str(parent_id) if parent_id else None
        if not updates:
            return resource_item(resource)
        updated = await self.request(
            "PUT",
            self.path(key, "resources", resource.id),
            json=content(raw, **updates, expected_version=resource.version),
        )
        return resource_item(parse(updated))

    async def _update_asset(
        self, key: str, asset: Asset, changes: dict[str, Any]
    ) -> SpaceItem:
        if not changes:
            return asset_item(asset)
        body: dict[str, Any] = {
            "title": changes.get("title", asset.title),
            "tags": asset.tags,
            "sources": [source.model_dump() for source in asset.sources],
            "expected_version": asset.version,
        }
        # Omitted pin and folder fields keep their saved values in Tsuchi.
        if "pinned" in changes:
            body["pinned"] = changes["pinned"]
        if "parent_id" in changes:
            folder_id = changes["parent_id"]
            body["folder_id"] = str(folder_id) if folder_id else None
        updated = await self.request(
            "PUT", self.path(key, "assets", asset.id), json=body
        )
        return asset_item(self._parse_asset(updated))

    async def delete(self, key: str, item_id: UUID) -> None:
        found = await self.find(key, item_id)
        if isinstance(found, Asset):
            await self._delete_asset(key, found)
            return
        for _ in range(WRITE_ATTEMPTS):
            try:
                await self._trash_tree(key, item_id)
                return
            except Retry:
                continue
        raise ServiceError(TRASHED_WITH_FOLDER, 409, "version_conflict")

    async def _delete_asset(self, key: str, asset: Asset) -> None:
        for attempt in range(WRITE_ATTEMPTS):
            if attempt:
                current = await self._read_asset(key, asset.id)
                if current is None or current.deleted_at is not None:
                    return
                asset = current
            try:
                await self._trash_asset(key, asset)
                return
            except Retry:
                continue
        raise ServiceError(BUSY, 409, "version_conflict")

    async def _trash_asset(self, key: str, asset: Asset) -> None:
        try:
            await self.request(
                "DELETE",
                self.path(key, "assets", asset.id),
                params={"expected_version": asset.version},
            )
        except ServiceError as error:
            if error.status == 409 and "Remove this asset" in str(error):
                raise ServiceError(IN_USE, 409, "file_in_use") from error
            if error.status == 409:
                raise Retry from error
            raise

    async def _trash_tree(self, key: str, root_id: UUID) -> None:
        """Trash everything inside first, because Tsuchi keeps parents of active items."""
        resources, assets = await asyncio.gather(
            self._resources(key, trash=False), self._assets(key, trash=False)
        )
        by_id = {resource.id: resource for resource in resources}
        if root_id not in by_id:
            return  # Already trashed by a concurrent request.
        children: dict[UUID, list[Resource]] = defaultdict(list)
        for resource in resources:
            if resource.parent_id is not None:
                children[resource.parent_id].append(resource)
        files: dict[UUID, list[Asset]] = defaultdict(list)
        for asset in assets:
            if asset.folder_id is not None:
                files[asset.folder_id].append(asset)

        order: list[Resource] = []
        visited: set[UUID] = set()

        def visit(resource: Resource) -> None:
            if resource.id in visited:
                return
            visited.add(resource.id)
            for child in children[resource.id]:
                visit(child)
            order.append(resource)

        visit(by_id[root_id])
        for resource in order:
            for asset in files[resource.id]:
                await self._trash_folder_file(key, asset)
            await self._trash_resource(key, resource)

    async def _trash_folder_file(self, key: str, asset: Asset) -> None:
        try:
            await self._trash_asset(key, asset)
        except ServiceError as error:
            if error.code != "file_in_use":
                raise
            # A file still used elsewhere stays, moved to the top level.
            try:
                await self.request(
                    "PUT",
                    self.path(key, "assets", asset.id),
                    json={
                        "title": asset.title,
                        "tags": asset.tags,
                        "sources": [source.model_dump() for source in asset.sources],
                        "folder_id": None,
                        "expected_version": asset.version,
                    },
                )
            except ServiceError as move_error:
                if move_error.status == 409:
                    raise Retry from move_error
                raise

    async def _trash_resource(self, key: str, resource: Resource) -> None:
        try:
            await self.request(
                "DELETE",
                self.path(key, "resources", resource.id),
                params={"expected_version": resource.version},
            )
        except ServiceError as error:
            if error.status == 404:
                return
            if error.status != 409:
                raise
            if "Move or trash" in str(error):
                await self._move_out_other_children(key, resource.id)
            raise Retry from error

    async def _move_out_other_children(self, key: str, parent_id: UUID) -> None:
        """Memories, personas or chats stored under a page move to the top level."""
        for other in await self._resources(key, trash=False, kinds=OTHER_KINDS):
            if other.parent_id != parent_id:
                continue
            found = await self._read_resource(key, other.id)
            if found is None:
                continue
            current, raw = found
            try:
                await self.request(
                    "PUT",
                    self.path(key, "resources", other.id),
                    json=content(raw, parent_id=None, expected_version=current.version),
                )
            except ServiceError as error:
                if error.status != 409:
                    raise

    async def restore(self, key: str, item_id: UUID) -> SpaceItem:
        for _ in range(WRITE_ATTEMPTS):
            found = await self.find(key, item_id, trashed=None)
            if isinstance(found, Asset):
                if found.deleted_at is None:
                    return asset_item(found)
                path = self.path(key, "assets", item_id, "restore")
                version = found.version
            else:
                resource, _ = found
                if resource.deleted_at is None:
                    return resource_item(resource)
                if resource.parent_id is not None:
                    await self._require_restored_parent(key, resource.parent_id)
                path = self.path(key, "resources", item_id, "restore")
                version = resource.version
            try:
                restored = await self.request(
                    "POST", path, json={"expected_version": version}
                )
            except ServiceError as error:
                detail = str(error)
                if error.status == 400 and "Parent must be" in detail:
                    raise ServiceError(RESTORE_FOLDER, 409, "parent_trashed") from error
                if error.status == 400 and "Restore attached assets" in detail:
                    raise ServiceError(RESTORE_FILES, 409, "files_trashed") from error
                if error.status == 409:
                    continue
                raise
            if isinstance(found, Asset):
                return asset_item(self._parse_asset(restored))
            return resource_item(parse(restored))
        raise ServiceError(BUSY, 409, "version_conflict")

    async def _require_restored_parent(self, key: str, parent_id: UUID) -> None:
        found = await self._read_resource(key, parent_id)
        if found is None:
            raise ServiceError(RESTORE_FOLDER, 409, "parent_trashed")
        parent, _ = found
        if parent.deleted_at is not None:
            message = (
                RESTORE_PAGE if parent.kind == ResourceKind.SPACE else RESTORE_FOLDER
            )
            raise ServiceError(message, 409, "parent_trashed")
