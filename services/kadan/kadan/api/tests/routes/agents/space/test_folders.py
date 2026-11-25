from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from kadan.api.config import ServiceError
from kadan.api.routes.agents.space.folders import FolderCreate, create_folder


def test_create_folder_at_the_top_and_inside_a_folder(tsuchi, call):
    top = call(create_folder, "alpha", FolderCreate(title=" Trips "))
    assert (top.type, top.title, top.parent_id, top.pinned) == (
        "folder",
        "Trips",
        None,
        False,
    )
    assert tsuchi.resources[str(top.id)]["kind"] == "folder"
    inner = call(create_folder, "alpha", FolderCreate(title="Japan", parent_id=top.id))
    assert inner.parent_id == top.id


def test_folder_cannot_go_inside_a_page(tsuchi, call):
    page = tsuchi.add("space", "Guide")
    with pytest.raises(ServiceError, match="only go inside folders") as error:
        call(create_folder, "alpha", FolderCreate(title="X", parent_id=page["id"]))
    assert error.value.status == 422
    assert tsuchi.writes("POST") == []


def test_folder_inside_a_missing_folder_is_refused(tsuchi, call):
    with pytest.raises(ServiceError, match="folder you chose no longer exists"):
        call(create_folder, "alpha", FolderCreate(title="X", parent_id=uuid4()))
    trashed = tsuchi.add("folder", deleted_at="2026-11-02T00:00:00+00:00")
    with pytest.raises(ServiceError, match="folder you chose no longer exists"):
        call(
            create_folder,
            "alpha",
            FolderCreate(title="X", parent_id=UUID(trashed["id"])),
        )


@pytest.mark.parametrize("body", [{"title": " "}, {}, {"title": "A", "kind": "x"}])
def test_folder_requires_a_name(body):
    with pytest.raises(ValidationError):
        FolderCreate.model_validate(body)
