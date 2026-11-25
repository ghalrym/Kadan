from uuid import UUID, uuid4

import pytest
from fastapi import Response
from pydantic import ValidationError

from kadan.api.config import ServiceError
from kadan.api.routes.agents.space.items import (
    ItemUpdate,
    delete_item,
    list_items,
    restore_item,
    update_item,
)

TRASHED = "2026-11-02T00:00:00+00:00"


def update(**body):
    return ItemUpdate.model_validate(body)


def test_list_shows_active_items_newest_first(tsuchi, call):
    folder = tsuchi.add("folder", "Trips")
    page = tsuchi.add("space", "Packing", parent_id=folder["id"], pinned=True)
    deck = tsuchi.add("deck", "Pitch")
    image = tsuchi.file("Beach", is_image=True, folder_id=folder["id"])
    file = tsuchi.file("Receipt")
    tsuchi.add("memory", "Likes tea")
    tsuchi.add("persona", "Helper")
    tsuchi.add("conversation", "Chat")
    tsuchi.add("space", "Old", deleted_at=TRASHED)
    tsuchi.file("Gone", deleted_at=TRASHED)
    tsuchi.file("Pet idle", tags=["pet", str(uuid4())])

    items = call(list_items, "alpha")

    assert [(item.title, item.type) for item in items] == [
        ("Receipt", "file"),
        ("Beach", "image"),
        ("Pitch", "slides"),
        ("Packing", "page"),
        ("Trips", "folder"),
    ]
    by_id = {str(item.id): item for item in items}
    assert by_id[page["id"]].parent_id == UUID(folder["id"])
    assert by_id[page["id"]].pinned is True
    assert by_id[image["id"]].parent_id == UUID(folder["id"])
    assert by_id[file["id"]].filename == "notes.txt"
    assert by_id[file["id"]].size_bytes == 12
    assert by_id[deck["id"]].deleted_at is None


def test_list_pages_through_resources_and_files(tsuchi, call):
    for _ in range(201):
        tsuchi.add("folder", "Folder")
        tsuchi.file()
    assert len(call(list_items, "alpha")) == 402
    offsets = [
        (path.rsplit("/", 1)[-1], params["offset"])
        for method, path, params, _ in tsuchi.requests
        if method == "GET"
    ]
    assert sorted(offsets) == [
        ("assets", "0"),
        ("assets", "200"),
        ("resources", "0"),
        ("resources", "200"),
    ]


@pytest.mark.parametrize(
    "body",
    [{"title": None}, {"title": " "}, {"pinned": None}, {"kind": "folder"}],
)
def test_update_rejects_null_blank_or_unknown_fields(body):
    with pytest.raises(ValidationError):
        update(**body)


def test_update_changes_only_fields_in_the_body(tsuchi, call):
    folder = tsuchi.add("folder", "Trips")
    page = tsuchi.add("space", "Packing", markdown="Socks", parent_id=folder["id"])
    page_id = UUID(page["id"])

    renamed = call(update_item, "alpha", page_id, update(title=" List "))
    assert (renamed.title, renamed.parent_id, renamed.pinned) == (
        "List",
        UUID(folder["id"]),
        False,
    )
    pinned = call(update_item, "alpha", page_id, update(pinned=True))
    assert (pinned.title, pinned.pinned) == ("List", True)
    moved = call(update_item, "alpha", page_id, update(parent_id=None))
    assert (moved.parent_id, moved.pinned) == (None, True)
    assert page["markdown"] == "Socks"


def test_empty_update_does_not_write(tsuchi, call):
    page = tsuchi.add("space", "Packing")
    file = tsuchi.file()
    assert call(update_item, "alpha", UUID(page["id"]), update()).version == 1
    assert call(update_item, "alpha", UUID(file["id"]), update()).version == 1
    assert tsuchi.writes("PUT") == []


def test_files_move_between_folders_and_keep_omitted_fields(tsuchi, call):
    folder = tsuchi.add("folder", "Trips")
    file = tsuchi.file(tags=["kept"], pinned=True)
    file_id = UUID(file["id"])

    moved = call(update_item, "alpha", file_id, update(parent_id=folder["id"]))
    assert (moved.parent_id, moved.pinned) == (UUID(folder["id"]), True)
    body = tsuchi.requests[-1][3]
    assert body == {
        "title": "Notes",
        "tags": ["kept"],
        "sources": [],
        "folder_id": folder["id"],
    }

    renamed = call(update_item, "alpha", file_id, update(title="Plan"))
    assert (renamed.title, renamed.parent_id) == ("Plan", UUID(folder["id"]))
    assert "folder_id" not in tsuchi.requests[-1][3]
    assert "pinned" not in tsuchi.requests[-1][3]

    to_top = call(update_item, "alpha", file_id, update(parent_id=None))
    assert to_top.parent_id is None and file["folder_id"] is None


@pytest.mark.parametrize("kind", ["space", "deck"])
def test_files_and_folders_go_only_inside_folders(tsuchi, call, kind):
    parent = tsuchi.add(kind, "Not a folder")
    for item in (tsuchi.file(), tsuchi.add("folder", "Folder")):
        with pytest.raises(ServiceError) as error:
            call(update_item, "alpha", UUID(item["id"]), update(parent_id=parent["id"]))
        assert error.value.status in {404, 422}
    assert tsuchi.writes("PUT") == []


def test_pages_can_go_inside_folders_or_pages(tsuchi, call):
    folder = tsuchi.add("folder", "Trips")
    parent = tsuchi.add("space", "Guide")
    page = tsuchi.add("space", "Day one")
    assert call(
        update_item, "alpha", UUID(page["id"]), update(parent_id=parent["id"])
    ).parent_id == UUID(parent["id"])
    slides = tsuchi.add("deck", "Pitch")
    assert call(
        update_item, "alpha", UUID(slides["id"]), update(parent_id=folder["id"])
    ).parent_id == UUID(folder["id"])


@pytest.mark.parametrize("parent", [None, {"deleted_at": TRASHED}, {"kind": "memory"}])
def test_moving_into_a_missing_folder_is_refused(tsuchi, call, parent):
    target = (
        tsuchi.add(**{"kind": "folder", **parent})["id"] if parent else str(uuid4())
    )
    page = tsuchi.add("space", "Page")
    with pytest.raises(
        ServiceError, match="folder you chose no longer exists"
    ) as error:
        call(update_item, "alpha", UUID(page["id"]), update(parent_id=target))
    assert error.value.status == 404


def test_a_folder_cannot_go_inside_itself(tsuchi, call):
    outer = tsuchi.add("folder", "Outer")
    inner = tsuchi.add("folder", "Inner", parent_id=outer["id"])
    with pytest.raises(ServiceError, match="inside itself") as error:
        call(update_item, "alpha", UUID(outer["id"]), update(parent_id=inner["id"]))
    assert error.value.status == 422


def test_update_retries_after_a_concurrent_change(tsuchi, call):
    page = tsuchi.add("space", "Page")
    edits = ["Edited elsewhere"]

    def edit(resource_id):
        if edits:
            record = tsuchi.resources[resource_id]
            record["markdown"] = edits.pop()
            record["version"] += 1

    tsuchi.before_write = edit
    item = call(update_item, "alpha", UUID(page["id"]), update(pinned=True))
    assert item.pinned is True and page["markdown"] == "Edited elsewhere"
    assert len(tsuchi.writes("PUT")) == 2


@pytest.mark.parametrize(
    "record",
    [
        {"kind": "memory"},
        {"kind": "conversation"},
        {"kind": "space", "deleted_at": TRASHED},
    ],
)
def test_non_items_and_trashed_items_are_missing(tsuchi, call, record):
    item_id = UUID(tsuchi.add(**record)["id"])
    for route, args in (
        (update_item, (update(title="Hi"),)),
        (delete_item, ()),
    ):
        with pytest.raises(ServiceError, match="This item no longer exists") as error:
            call(route, "alpha", item_id, *args)
        assert error.value.status == 404


def test_unknown_ids_and_pet_artwork_are_missing(tsuchi, call):
    pet = tsuchi.file("Pet idle", tags=["pet"])
    for item_id in (uuid4(), UUID(pet["id"])):
        with pytest.raises(ServiceError, match="This item no longer exists"):
            call(update_item, "alpha", item_id, update(title="Hi"))


def test_delete_moves_a_file_to_trash(tsuchi, call):
    file = tsuchi.file()
    response = call(delete_item, "alpha", UUID(file["id"]))
    assert isinstance(response, Response) and response.status_code == 204
    assert file["deleted_at"] is not None


def test_a_file_in_use_cannot_be_trashed(tsuchi, call):
    file = tsuchi.file()
    tsuchi.in_use.add(file["id"])
    with pytest.raises(ServiceError, match="attached to a chat or page") as error:
        call(delete_item, "alpha", UUID(file["id"]))
    assert error.value.status == 409 and file["deleted_at"] is None


def test_delete_trashes_a_folder_with_everything_inside_it(tsuchi, call):
    root = tsuchi.add("folder", "Trips")
    sub = tsuchi.add("folder", "Japan", parent_id=root["id"])
    page = tsuchi.add("space", "Guide", parent_id=sub["id"])
    subpage = tsuchi.add("space", "Day one", parent_id=page["id"])
    deck = tsuchi.add("deck", "Slides", parent_id=root["id"])
    photo = tsuchi.file("Photo", is_image=True, folder_id=sub["id"])
    receipt = tsuchi.file("Receipt", folder_id=root["id"])
    outside = tsuchi.add("space", "Elsewhere")

    call(delete_item, "alpha", UUID(root["id"]))

    for record in (root, sub, page, subpage, deck, photo, receipt):
        assert record["deleted_at"] is not None, record["title"]
    assert outside["deleted_at"] is None
    trashed = [path.rsplit("/", 1)[-1] for path in tsuchi.writes("DELETE")]
    assert trashed.index(subpage["id"]) < trashed.index(page["id"])
    assert trashed.index(photo["id"]) < trashed.index(sub["id"])
    assert trashed[-1] == root["id"]


def test_a_used_file_leaves_its_folder_instead_of_blocking_the_delete(tsuchi, call):
    folder = tsuchi.add("folder", "Trips")
    used = tsuchi.file("Attached", folder_id=folder["id"])
    tsuchi.in_use.add(used["id"])
    call(delete_item, "alpha", UUID(folder["id"]))
    assert folder["deleted_at"] is not None
    assert (used["deleted_at"], used["folder_id"]) == (None, None)


def test_other_records_under_a_page_move_to_the_top(tsuchi, call):
    page = tsuchi.add("space", "Guide")
    memory = tsuchi.add("memory", "Likes tea", parent_id=page["id"])
    call(delete_item, "alpha", UUID(page["id"]))
    assert page["deleted_at"] is not None
    assert (memory["deleted_at"], memory["parent_id"]) == (None, None)


def test_restore_brings_back_a_trashed_item(tsuchi, call):
    folder = tsuchi.add("folder", "Trips", deleted_at=TRASHED)
    restored = call(restore_item, "alpha", UUID(folder["id"]))
    assert restored.deleted_at is None and folder["deleted_at"] is None
    # Restoring again is harmless.
    assert call(restore_item, "alpha", UUID(folder["id"])).version == restored.version


@pytest.mark.parametrize(
    ("parent_kind", "message"),
    [
        ("folder", "Restore the folder it was in first."),
        ("space", "Restore the page it was in first."),
    ],
)
def test_restore_inside_a_trashed_parent_is_refused(tsuchi, call, parent_kind, message):
    parent = tsuchi.add(parent_kind, "Parent", deleted_at=TRASHED)
    page = tsuchi.add("space", "Guide", parent_id=parent["id"], deleted_at=TRASHED)
    with pytest.raises(ServiceError) as error:
        call(restore_item, "alpha", UUID(page["id"]))
    assert (error.value.status, str(error.value)) == (409, message)
    assert page["deleted_at"] == TRASHED


def test_restore_reports_a_parent_trashed_during_the_request(tsuchi, call):
    parent = tsuchi.add("folder", "Parent")
    page = tsuchi.add("space", "Guide", parent_id=parent["id"], deleted_at=TRASHED)

    def trash_parent(_):
        parent["deleted_at"] = TRASHED

    tsuchi.before_write = trash_parent
    with pytest.raises(
        ServiceError, match="Restore the folder it was in first"
    ) as error:
        call(restore_item, "alpha", UUID(page["id"]))
    assert error.value.status == 409


def test_restored_file_returns_to_the_top_when_its_folder_is_gone(tsuchi, call):
    folder = tsuchi.add("folder", "Trips", deleted_at=TRASHED)
    file = tsuchi.file(folder_id=folder["id"], deleted_at=TRASHED)
    restored = call(restore_item, "alpha", UUID(file["id"]))
    assert (restored.deleted_at, restored.parent_id) == (None, None)
