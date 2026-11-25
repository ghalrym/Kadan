"""An in-memory Tsuchi with resources and assets for the Space routes."""

import json
from uuid import uuid4

import httpx
import pytest

from ..conftest import FakeTsuchi

ROOT = "/api/agents/alpha"
VERSION_CONFLICT = {"detail": "This item changed. Reload it before saving again."}
PARENT_KINDS = {
    "folder": {"folder"},
    "space": {"folder", "space"},
    "deck": {"folder", "space"},
}
PARENT_ERRORS = {
    "folder": "Parent must be an active folder in this namespace",
    "space": "Parent must be an active folder or Space in this namespace",
    "deck": "Parent must be an active folder or Space in this namespace",
}


def error(status, detail):
    return httpx.Response(status, json={"detail": detail})


class FakeSpaceTsuchi(FakeTsuchi):
    def __init__(self):
        super().__init__()
        self.assets: dict[str, dict] = {}
        # Asset IDs that active resources still reference.
        self.in_use: set[str] = set()

    def add(self, kind="conversation", title="New chat", **changes) -> dict:
        return super().add(kind, title, **{"pinned": False, **changes})

    def file(self, title="Notes", is_image=False, **changes) -> dict:
        now = self.tick()
        record = {
            "id": str(uuid4()),
            "agent_key": "alpha",
            "title": title,
            "filename": "photo.png" if is_image else "notes.txt",
            "media_type": "image/png" if is_image else "text/plain",
            "size_bytes": 12,
            "sha256": "a" * 64,
            "is_image": is_image,
            "tags": [],
            "sources": [],
            "pinned": False,
            "folder_id": None,
            "version": 1,
            "created_at": now,
            "updated_at": now,
            "deleted_at": None,
            **changes,
        }
        self.assets[record["id"]] = record
        return record

    def bump(self, record, **changes):
        record.update(changes)
        record["version"] += 1
        record["updated_at"] = self.tick()
        return httpx.Response(200, json=record)

    def active_folder(self, folder_id) -> bool:
        folder = self.resources.get(folder_id)
        return bool(folder and folder["kind"] == "folder" and not folder["deleted_at"])

    def parent_error(self, record):
        parent_id = record.get("parent_id")
        if not parent_id:
            return None
        parent = self.resources.get(parent_id)
        if parent is None:
            return error(404, "Resource not found in this namespace")
        allowed = PARENT_KINDS.get(record["kind"], {"space"})
        if parent["kind"] not in allowed or parent["deleted_at"]:
            return error(400, PARENT_ERRORS.get(record["kind"], "Parent must be"))
        ancestor, seen = parent, {record["id"]}
        while ancestor:
            if ancestor["id"] in seen:
                return error(
                    400,
                    "A folder or Space cannot contain itself or one of its ancestors",
                )
            seen.add(ancestor["id"])
            ancestor = self.resources.get(ancestor["parent_id"] or "")
        return None

    def page(self, rows, params):
        rows.sort(key=lambda record: record["id"])
        rows.sort(key=lambda record: record["updated_at"], reverse=True)
        offset, limit = int(params["offset"]), int(params["limit"])
        return httpx.Response(200, json=rows[offset : offset + limit])

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path, method, params = request.url.path, request.method, request.url.params
        body = json.loads(request.content) if request.content else None
        self.requests.append((method, path, dict(params), body))
        trash = params.get("trash") == "true"
        if path == f"{ROOT}/resources" and method == "GET":
            kinds = params.get_list("kinds")
            return self.page(
                [
                    record
                    for record in self.resources.values()
                    if bool(record["deleted_at"]) == trash and record["kind"] in kinds
                ],
                params,
            )
        if path == f"{ROOT}/resources" and method == "POST":
            record = self.add(**body)
            failure = self.parent_error(record)
            if failure is not None:
                del self.resources[record["id"]]
                return failure
            return httpx.Response(201, json=record)
        if path == f"{ROOT}/assets" and method == "GET":
            return self.page(
                [r for r in self.assets.values() if bool(r["deleted_at"]) == trash],
                params,
            )
        segments = path.removeprefix(ROOT + "/").split("/")
        collection, item_id, action = (segments + [None, None])[:3]
        store = self.resources if collection == "resources" else self.assets
        record = store.get(item_id)
        if record is None:
            name = "Resource" if collection == "resources" else "Asset"
            return error(404, f"{name} not found in this namespace")
        if method == "GET":
            return httpx.Response(200, json=record)
        if self.before_write is not None:
            self.before_write(record["id"])
        if method == "PUT":
            expected = body.pop("expected_version")
        elif method == "POST":
            expected = body["expected_version"]
        else:
            expected = int(params["expected_version"])
        if expected != record["version"]:
            return httpx.Response(409, json=VERSION_CONFLICT)
        if action == "restore":
            if not record["deleted_at"]:
                return error(409, "This item is not in trash")
            if collection == "assets":
                folder_id = record["folder_id"]
                if folder_id and not self.active_folder(folder_id):
                    folder_id = None
                return self.bump(record, deleted_at=None, folder_id=folder_id)
            failure = self.parent_error(record)
            if failure is not None:
                return failure
            return self.bump(record, deleted_at=None)
        if record["deleted_at"]:
            return error(409, "Restore this item from trash before editing it")
        if method == "PUT" and collection == "assets":
            if body.get("folder_id") and not self.active_folder(body["folder_id"]):
                return error(400, "Folder must be an active folder in this namespace")
            return self.bump(record, **body)
        if method == "PUT":
            candidate = {**record, **body}
            failure = self.parent_error(candidate)
            if failure is not None:
                return failure
            return self.bump(record, **body)
        if collection == "assets":
            if item_id in self.in_use:
                return error(
                    409,
                    "Remove this asset from active resources before moving it to trash",
                )
            return self.bump(record, deleted_at=self.tick())
        children = any(
            child["parent_id"] == item_id and not child["deleted_at"]
            for child in self.resources.values()
        ) or any(
            asset["folder_id"] == item_id and not asset["deleted_at"]
            for asset in self.assets.values()
        )
        if children:
            return error(409, "Move or trash this folder's contents first")
        return self.bump(record, deleted_at=self.tick())


@pytest.fixture
def tsuchi():
    return FakeSpaceTsuchi()
