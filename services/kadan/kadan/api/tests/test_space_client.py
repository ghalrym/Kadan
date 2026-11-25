from datetime import UTC, datetime
from uuid import uuid4

import pytest

from kadan.api.config import ServiceError
from kadan.api.pydantic_models.assets import Asset
from kadan.api.pydantic_models.resources import Resource
from kadan.api.space_client import (
    asset_item,
    is_pet_artwork,
    plain_write_error,
    resource_item,
)

NOW = datetime(2026, 11, 1, tzinfo=UTC)


def resource(kind, **changes):
    return Resource(
        id=uuid4(),
        agent_key="alpha",
        kind=kind,
        title="Item",
        version=2,
        created_at=NOW,
        updated_at=NOW,
        **changes,
    )


def asset(**changes):
    fields = {
        "id": uuid4(),
        "agent_key": "alpha",
        "title": "Photo",
        "filename": "photo.png",
        "media_type": "image/png",
        "size_bytes": 10,
        "sha256": "a" * 64,
        "is_image": True,
        "version": 1,
        "created_at": NOW,
        "updated_at": NOW,
        "deleted_at": None,
    }
    return Asset(**{**fields, **changes})


@pytest.mark.parametrize(
    ("kind", "shown"), [("folder", "folder"), ("space", "page"), ("deck", "slides")]
)
def test_resources_map_to_item_types(kind, shown):
    parent_id = uuid4()
    item = resource_item(resource(kind, parent_id=parent_id, pinned=True))
    assert (item.type, item.parent_id, item.pinned, item.version) == (
        shown,
        parent_id,
        True,
        2,
    )
    assert (item.filename, item.media_type, item.size_bytes) == (None, None, None)


def test_assets_map_to_images_and_files_in_their_folder():
    folder_id = uuid4()
    image = asset_item(asset(folder_id=folder_id))
    assert (image.type, image.parent_id, image.filename, image.size_bytes) == (
        "image",
        folder_id,
        "photo.png",
        10,
    )
    file = asset_item(asset(is_image=False, media_type="text/plain"))
    assert (file.type, file.parent_id, file.pinned) == ("file", None, False)


def test_records_without_pin_or_folder_load_with_defaults():
    assert resource("space").pinned is False
    stored = asset()
    assert (stored.pinned, stored.folder_id) == (False, None)


def test_pet_artwork_is_recognized_by_its_tag():
    assert is_pet_artwork(asset(tags=["pet", str(uuid4())]))
    assert not is_pet_artwork(asset(tags=["generated-image"]))


@pytest.mark.parametrize(
    ("status", "detail", "shown", "code"),
    [
        (
            400,
            "A folder or Space cannot contain itself or one of its ancestors",
            422,
            "invalid_parent",
        ),
        (400, "Parent must be an active folder in this namespace", 404, "not_found"),
        (400, "Folder must be an active folder in this namespace", 404, "not_found"),
        (404, "Resource not found in this namespace", 404, "not_found"),
    ],
)
def test_link_rejections_are_reworded(status, detail, shown, code):
    reworded = plain_write_error(ServiceError(detail, status))
    assert (reworded.status, reworded.code) == (shown, code)
    assert "Space" not in str(reworded) and "namespace" not in str(reworded)


def test_other_errors_pass_through():
    original = ServiceError("Tsuchi is unavailable.", 503)
    assert plain_write_error(original) is original
