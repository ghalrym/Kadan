from uuid import UUID

import pytest
from pydantic import ValidationError

from kadan.api.config import ServiceError
from kadan.api.routes.agents.spaces import (
    SpaceCreate,
    SpaceUpdate,
    create_space,
    delete_space,
    get_space,
    list_spaces,
    update_space,
)


@pytest.mark.parametrize(
    ("model", "body"),
    [
        (SpaceCreate, {"title": " "}),
        (SpaceCreate, {"title": "Garden", "instructions": "x" * 48001}),
        (SpaceCreate, {"title": "Garden", "parent_id": None}),
        (SpaceUpdate, {"title": "Garden", "instructions": ""}),
        (SpaceUpdate, {"title": "Garden", "expected_version": 1}),
    ],
)
def test_space_requests_reject_invalid_input(model, body):
    with pytest.raises(ValidationError):
        model.model_validate(body)


def test_create_list_and_read_spaces(tsuchi, call):
    tsuchi.add("space", "Saved document")
    older = call(create_space, "alpha", SpaceCreate(title=" Garden "))
    newer = call(
        create_space,
        "alpha",
        SpaceCreate(title="Travel", instructions="Prefer trains."),
    )
    assert (older.title, older.instructions) == ("Garden", "")
    assert tsuchi.resources[str(newer.id)]["kind"] == "project"
    assert tsuchi.resources[str(newer.id)]["markdown"] == "Prefer trains."
    assert [space.id for space in call(list_spaces, "alpha")] == [newer.id, older.id]
    assert call(get_space, "alpha", newer.id) == newer


@pytest.mark.parametrize(
    "record",
    [{"kind": "space"}, {"kind": "project", "deleted_at": "2026-11-02T00:00:00Z"}],
)
def test_get_rejects_documents_and_trashed_spaces(tsuchi, call, record):
    space_id = UUID(tsuchi.add(**record)["id"])
    with pytest.raises(ServiceError, match="This Space no longer exists") as error:
        call(get_space, "alpha", space_id)
    assert error.value.status == 404


def test_update_keeps_other_fields_and_passes_version_conflicts(tsuchi, call):
    record = tsuchi.add("project", "Garden", tags=["kept"])
    space_id = UUID(record["id"])
    saved = call(
        update_space,
        "alpha",
        space_id,
        SpaceUpdate(title="Yard", instructions="Be brief.", expected_version=1),
    )
    assert (saved.title, saved.instructions, saved.version) == ("Yard", "Be brief.", 2)
    assert record["tags"] == ["kept"]
    with pytest.raises(ServiceError, match="This item changed") as error:
        call(
            update_space,
            "alpha",
            space_id,
            SpaceUpdate(title="Lawn", instructions="", expected_version=1),
        )
    assert error.value.status == 409
    assert record["title"] == "Yard"


def test_delete_trashes_the_space_chats_before_the_space(tsuchi, call):
    space = tsuchi.add("project", "Garden")
    inside = [tsuchi.add(parent_id=space["id"]) for _ in range(2)]
    outside = tsuchi.add()
    call(delete_space, "alpha", UUID(space["id"]))
    deletes = tsuchi.writes("DELETE")
    assert deletes[-1].endswith(space["id"])
    assert sorted(path.rsplit("/", 1)[1] for path in deletes[:-1]) == sorted(
        chat["id"] for chat in inside
    )
    assert all(chat["deleted_at"] for chat in [*inside, space])
    assert outside["deleted_at"] is None


def test_delete_also_trashes_a_chat_added_during_deletion(tsuchi, call):
    space = tsuchi.add("project", "Garden")
    late = []

    def chat_added(resource_id):
        if resource_id == space["id"] and not late:
            late.append(tsuchi.add(parent_id=space["id"]))

    tsuchi.before_write = chat_added
    call(delete_space, "alpha", UUID(space["id"]))
    assert space["deleted_at"] and late[0]["deleted_at"]
