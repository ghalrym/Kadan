"""Folders, asset folders and pinning against a temporary SQLite file."""

import json
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from pydantic import ValidationError
from tsuchi.config import StorageError
from tsuchi.models import (
    AgentInput,
    Asset,
    AssetMetadata,
    AssetUpdate,
    ConversationContent,
    DeckContent,
    MarkdownContent,
    MarkdownUpdate,
    ResourceKind,
    ResourceRecord,
    resource_adapter,
    resource_input_adapter,
    utc_now,
)
from tsuchi.repository import Repository
from tsuchi.routes import create_router


def content(resource, **changes):
    return resource_input_adapter.validate_python(
        {**resource.model_dump(exclude=set(ResourceRecord.model_fields)), **changes}
    )


def create_asset(repository):
    now = utc_now()
    return repository.create_asset(
        Asset(
            id=uuid4(),
            agent_key="alpha",
            title="File",
            filename="file.txt",
            media_type="text/plain",
            size_bytes=3,
            sha256="abc",
            is_image=False,
            version=1,
            created_at=now,
            updated_at=now,
        )
    )


@pytest.fixture
def repository(tmp_path):
    repository = Repository(tmp_path)
    repository.initialize()
    repository.create_agent(AgentInput(key="alpha", name="Alpha"))
    repository.create_agent(AgentInput(key="beta", name="Beta"))
    return repository


def folder(repository, title="Folder", parent_id=None, key="alpha"):
    return repository.create_resource(
        key, MarkdownContent(kind="folder", title=title, parent_id=parent_id)
    )


def page(repository, parent_id=None, kind="space"):
    if kind == "deck":
        return repository.create_resource(
            "alpha",
            DeckContent(kind="deck", title="Slides", deck={}, parent_id=parent_id),
        )
    return repository.create_resource(
        "alpha", MarkdownContent(kind=kind, title="Page", parent_id=parent_id)
    )


def test_folders_hold_no_text_or_attachments():
    with pytest.raises(ValidationError, match="Folders cannot hold"):
        MarkdownContent(kind="folder", title="Folder", markdown="text")
    with pytest.raises(ValidationError, match="Folders cannot hold"):
        MarkdownContent(kind="folder", title="Folder", attachment_ids=[uuid4()])


def test_folders_nest_and_hold_pages_and_slides(repository):
    outer = folder(repository)
    inner = folder(repository, "Inner", outer.id)
    in_folder = page(repository, inner.id)
    assert page(repository, inner.id, "deck").parent_id == inner.id
    assert page(repository, in_folder.id).parent_id == in_folder.id
    assert inner.parent_id == outer.id


@pytest.mark.parametrize("kind", ["memory", "persona"])
def test_only_folders_pages_and_slides_go_in_folders(repository, kind):
    parent = folder(repository)
    with pytest.raises(StorageError, match="active Space"):
        page(repository, parent.id, kind)
    with pytest.raises(StorageError, match="active Space"):
        repository.create_resource(
            "alpha",
            ConversationContent(kind="conversation", title="Chat", parent_id=parent.id),
        )


def test_folder_cannot_be_placed_in_a_page_or_a_trashed_folder(repository):
    parent = page(repository)
    with pytest.raises(StorageError, match="active folder in"):
        folder(repository, parent_id=parent.id)
    trashed = folder(repository)
    repository.trash_resource("alpha", trashed.id, trashed.version)
    with pytest.raises(StorageError, match="active folder in"):
        folder(repository, parent_id=trashed.id)


def test_folder_cannot_be_placed_in_another_agents_folder(repository):
    other = folder(repository, key="beta")
    with pytest.raises(StorageError) as error:
        folder(repository, parent_id=other.id)
    assert error.value.status_code == 404


def test_folder_cycle_is_rejected(repository):
    outer = folder(repository)
    inner = folder(repository, "Inner", outer.id)
    with pytest.raises(StorageError, match="ancestors"):
        repository.update_resource(
            "alpha", outer.id, content(outer, parent_id=inner.id), outer.version
        )


def test_memory_in_a_page_inside_a_folder_is_still_allowed():
    """Each ancestor is checked by the rules for its own kind."""
    repository = Repository.__new__(Repository)
    folder_id = uuid4()
    repository._resource = Mock(
        side_effect=[
            SimpleNamespace(
                kind=ResourceKind.SPACE, deleted_at=None, parent_id=folder_id
            ),
            SimpleNamespace(kind=ResourceKind.FOLDER, deleted_at=None, parent_id=None),
        ]
    )
    repository._validate_links(
        Mock(),
        "alpha",
        MarkdownContent(kind="memory", title="Note", parent_id=uuid4()),
        uuid4(),
    )
    assert repository._resource.call_count == 2


def test_asset_folder_is_validated_on_upload_and_update(repository):
    parent = folder(repository)
    not_folder = page(repository)
    now = utc_now()
    fields = {
        "agent_key": "alpha",
        "title": "File",
        "filename": "file.txt",
        "media_type": "text/plain",
        "size_bytes": 3,
        "sha256": "abc",
        "is_image": False,
        "version": 1,
        "created_at": now,
        "updated_at": now,
    }
    uploaded = repository.create_asset(Asset(id=uuid4(), folder_id=parent.id, **fields))
    assert uploaded.folder_id == parent.id
    with pytest.raises(StorageError, match="active folder"):
        repository.create_asset(Asset(id=uuid4(), folder_id=not_folder.id, **fields))

    asset = create_asset(repository)
    with pytest.raises(StorageError, match="active folder"):
        repository.update_asset(
            "alpha",
            asset.id,
            AssetMetadata(title="File", folder_id=not_folder.id),
            asset.version,
        )
    moved = repository.update_asset(
        "alpha", asset.id, AssetMetadata(title="File", folder_id=parent.id), 1
    )
    assert moved.folder_id == parent.id
    to_root = repository.update_asset(
        "alpha", asset.id, AssetMetadata(title="File", folder_id=None), 2
    )
    assert to_root.folder_id is None


def test_asset_update_keeps_omitted_pin_and_folder(repository):
    parent = folder(repository)
    asset = create_asset(repository)
    asset = repository.update_asset(
        "alpha",
        asset.id,
        AssetMetadata(title="File", folder_id=parent.id, pinned=True),
        asset.version,
    )
    renamed = repository.update_asset(
        "alpha", asset.id, AssetMetadata(title="Renamed"), asset.version
    )
    assert (renamed.title, renamed.folder_id, renamed.pinned) == (
        "Renamed",
        parent.id,
        True,
    )


def endpoint(path, method):
    repository = Mock()
    router = create_router(repository, Mock())
    route = next(
        route
        for route in router.routes
        if route.path == path and method in route.methods
    )
    return repository, route.endpoint


def test_asset_update_route_leaves_omitted_pin_and_folder_unset():
    repository, update = endpoint("/api/agents/{key}/assets/{asset_id}", "PUT")
    asset_id, folder_id = uuid4(), uuid4()
    update("alpha", asset_id, AssetUpdate(title="Renamed", expected_version=3))
    update(
        "alpha",
        asset_id,
        AssetUpdate(title="Moved", folder_id=folder_id, expected_version=4),
    )
    (_, _, renamed, version), _ = repository.update_asset.call_args_list[0]
    assert version == 3 and renamed.title == "Renamed"
    assert renamed.model_fields_set.isdisjoint({"pinned", "folder_id"})
    (_, _, moved, _), _ = repository.update_asset.call_args_list[1]
    assert moved.folder_id == folder_id and "folder_id" in moved.model_fields_set
    assert "pinned" not in moved.model_fields_set


def test_upload_route_accepts_a_folder():
    files = Mock()
    router = create_router(Mock(), files)
    upload = next(
        route.endpoint
        for route in router.routes
        if route.path == "/api/agents/{key}/assets" and "POST" in route.methods
    )
    folder_id = uuid4()
    file = SimpleNamespace(file=Mock(), filename="notes.txt")
    upload("alpha", file, folder_id=folder_id)
    metadata = files.upload.call_args.args[3]
    assert (metadata.title, metadata.folder_id) == ("notes.txt", folder_id)


def test_restored_asset_returns_to_its_folder_or_to_the_top(repository):
    parent = folder(repository)
    asset = create_asset(repository)
    asset = repository.update_asset(
        "alpha", asset.id, AssetMetadata(title="File", folder_id=parent.id), 1
    )
    asset = repository.trash_asset("alpha", asset.id, asset.version)
    restored = repository.restore_asset("alpha", asset.id, asset.version)
    assert restored.folder_id == parent.id

    asset = repository.trash_asset("alpha", restored.id, restored.version)
    current = repository.get_resource("alpha", parent.id)
    repository.trash_resource("alpha", parent.id, current.version)
    restored = repository.restore_asset("alpha", asset.id, asset.version)
    assert restored.folder_id is None and restored.deleted_at is None


def test_folder_with_an_active_file_cannot_be_trashed(repository):
    parent = folder(repository)
    asset = create_asset(repository)
    asset = repository.update_asset(
        "alpha", asset.id, AssetMetadata(title="File", folder_id=parent.id), 1
    )
    with pytest.raises(StorageError, match="folder's contents") as error:
        repository.trash_resource("alpha", parent.id, parent.version)
    assert error.value.status_code == 409
    repository.trash_asset("alpha", asset.id, asset.version)
    assert repository.trash_resource("alpha", parent.id, parent.version).deleted_at


def test_resource_pin_round_trips_and_is_kept_when_omitted(repository):
    created = repository.create_resource(
        "alpha", MarkdownContent(kind="space", title="Page", pinned=True)
    )
    assert created.pinned is True
    # An editor that does not know about pins sends no pinned field.
    edited = repository.update_resource(
        "alpha",
        created.id,
        MarkdownContent(kind="space", title="Edited"),
        created.version,
    )
    assert (edited.title, edited.pinned) == ("Edited", True)
    unpinned = repository.update_resource(
        "alpha", created.id, content(edited, pinned=False), edited.version
    )
    assert unpinned.pinned is False
    restored = repository.restore_version("alpha", created.id, 1, unpinned.version)
    assert (restored.title, restored.pinned) == ("Page", False)


def test_resource_update_route_leaves_omitted_pin_unset():
    repository, update = endpoint("/api/agents/{key}/resources/{resource_id}", "PUT")
    update(
        "alpha",
        uuid4(),
        MarkdownUpdate(kind="space", title="Page", expected_version=1),
    )
    update(
        "alpha",
        uuid4(),
        MarkdownUpdate(kind="space", title="Page", pinned=True, expected_version=1),
    )
    (_, _, unpinned, _), _ = repository.update_resource.call_args_list[0]
    assert "pinned" not in unpinned.model_fields_set
    (_, _, pinned, _), _ = repository.update_resource.call_args_list[1]
    assert pinned.pinned is True and "pinned" in pinned.model_fields_set


def test_records_saved_before_pinning_load_unpinned():
    now = utc_now().isoformat()
    record = {
        "id": str(uuid4()),
        "agent_key": "alpha",
        "kind": "space",
        "title": "Old",
        "version": 1,
        "created_at": now,
        "updated_at": now,
    }
    assert resource_adapter.validate_json(json.dumps(record)).pinned is False
    del record["kind"]
    asset = Asset.model_validate(
        {
            **record,
            "filename": "a.txt",
            "media_type": "text/plain",
            "size_bytes": 1,
            "sha256": "abc",
            "is_image": False,
        }
    )
    assert (asset.pinned, asset.folder_id) == (False, None)
