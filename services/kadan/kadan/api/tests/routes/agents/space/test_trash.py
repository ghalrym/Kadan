from kadan.api.routes.agents.space.trash import list_trash

TRASHED = "2026-11-02T00:00:00+00:00"


def test_trash_lists_only_trashed_items_newest_first(tsuchi, call):
    tsuchi.add("space", "Kept")
    tsuchi.file("Kept file")
    folder = tsuchi.add("folder", "Old folder", deleted_at=TRASHED)
    tsuchi.file("Old file", deleted_at=TRASHED, folder_id=folder["id"])
    tsuchi.add("memory", "Forgotten", deleted_at=TRASHED)
    tsuchi.file("Old pet", deleted_at=TRASHED, tags=["pet"])

    items = call(list_trash, "alpha")

    assert [(item.title, item.type) for item in items] == [
        ("Old file", "file"),
        ("Old folder", "folder"),
    ]
    assert all(item.deleted_at is not None for item in items)
    assert items[0].parent_id == items[1].id
