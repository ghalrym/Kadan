from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from kadan.api.config import ServiceError
from kadan.api.routes.agents.space.pages import (
    PageCreate,
    PageUpdate,
    create_page,
    get_page,
    update_page,
)


def test_create_and_read_a_page(tsuchi, call):
    folder = tsuchi.add("folder", "Trips")
    created = call(
        create_page,
        "alpha",
        PageCreate(title=" Packing ", markdown="# Socks", parent_id=folder["id"]),
    )
    assert (created.type, created.title, created.parent_id) == (
        "page",
        "Packing",
        UUID(folder["id"]),
    )
    assert tsuchi.resources[str(created.id)]["kind"] == "space"
    page = call(get_page, "alpha", created.id)
    assert (page.markdown, page.version) == ("# Socks", 1)
    empty = call(create_page, "alpha", PageCreate(title="Blank"))
    assert call(get_page, "alpha", empty.id).markdown == ""


def test_pages_can_be_subpages(tsuchi, call):
    parent = tsuchi.add("space", "Guide")
    child = call(create_page, "alpha", PageCreate(title="Day", parent_id=parent["id"]))
    assert child.parent_id == UUID(parent["id"])


def test_page_inside_slides_is_refused(tsuchi, call):
    deck = tsuchi.add("deck", "Pitch")
    with pytest.raises(ServiceError, match="folder you chose no longer exists"):
        call(create_page, "alpha", PageCreate(title="X", parent_id=deck["id"]))


@pytest.mark.parametrize(
    "record",
    [
        {"kind": "deck"},
        {"kind": "memory"},
        {"kind": "space", "deleted_at": "2026-11-02T00:00:00+00:00"},
    ],
)
def test_only_active_pages_are_read(tsuchi, call, record):
    item_id = UUID(tsuchi.add(**record)["id"])
    with pytest.raises(ServiceError, match="This item no longer exists") as error:
        call(get_page, "alpha", item_id)
    assert error.value.status == 404
    with pytest.raises(ServiceError, match="This item no longer exists"):
        call(get_page, "alpha", uuid4())


def test_update_saves_text_and_keeps_other_fields(tsuchi, call):
    folder = tsuchi.add("folder", "Trips")
    page = tsuchi.add(
        "space", "Packing", parent_id=folder["id"], pinned=True, tags=["kept"]
    )
    saved = call(
        update_page,
        "alpha",
        UUID(page["id"]),
        PageUpdate(title="Packing list", markdown="Socks", expected_version=1),
    )
    assert (saved.title, saved.markdown, saved.version) == ("Packing list", "Socks", 2)
    assert (saved.parent_id, saved.pinned) == (UUID(folder["id"]), True)
    assert page["tags"] == ["kept"]


def test_update_reports_a_conflict_without_saving(tsuchi, call):
    page = tsuchi.add("space", "Packing", markdown="Newer", version=3)
    with pytest.raises(ServiceError, match="This item changed") as error:
        call(
            update_page,
            "alpha",
            UUID(page["id"]),
            PageUpdate(title="Packing", markdown="Older", expected_version=2),
        )
    assert error.value.status == 409
    assert page["markdown"] == "Newer"


@pytest.mark.parametrize(
    "body",
    [
        {"title": "A", "markdown": "B"},
        {"title": " ", "markdown": "", "expected_version": 1},
        {"title": "A", "markdown": "", "expected_version": 0},
    ],
)
def test_update_requires_a_title_text_and_version(body):
    with pytest.raises(ValidationError):
        PageUpdate.model_validate(body)
