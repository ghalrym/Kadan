from uuid import UUID, uuid4

import pytest
from fastapi import Response
from pydantic import ValidationError

from kadan.api.config import ServiceError
from kadan.api.routes.agents.conversations import (
    ConversationCreate,
    ConversationUpdate,
    create_conversation,
    delete_conversation,
    list_conversations,
    update_conversation,
)


def update(**body):
    return ConversationUpdate.model_validate(body)


def test_list_summarizes_active_chats_newest_first(tsuchi, call):
    space = tsuchi.add("project", "Garden")
    legacy = tsuchi.add("space", "Notes")
    old = tsuchi.add(title="Chain")
    tsuchi.message(old, "assistant", "Welcome")
    tsuchi.message(old, "user", "  Plan   the\n spring " + "beds " * 40)
    named = tsuchi.add(title="Seed order", parent_id=space["id"])
    tsuchi.message(named, "user", "Which seeds?")
    tsuchi.add(title="Filed", parent_id=legacy["id"])
    tsuchi.add(title="Trashed", deleted_at="2026-11-02T00:00:00+00:00")
    empty = tsuchi.add()

    summaries = call(list_conversations, "alpha")

    assert [item.title for item in summaries] == [
        "New chat",
        "Filed",
        "Seed order",
        "Plan the spring beds beds beds beds beds beds beds beds bed…",
    ]
    by_title = {item.title: item for item in summaries}
    assert by_title["Seed order"].space_id == UUID(space["id"])
    assert by_title["Seed order"].preview == "Which seeds?"
    assert by_title["Filed"].space_id is None
    assert by_title["New chat"].id == UUID(empty["id"])
    assert by_title["New chat"].preview == ""
    first = summaries[-1]
    assert len(first.title) == 60
    assert first.preview.startswith("Plan the spring beds")
    assert len(first.preview) == 160 and first.preview.endswith("…")
    assert "  " not in first.preview
    assert first.message_count == 2
    assert first.version == 3


def test_list_pages_through_every_chat(tsuchi, call):
    for _ in range(205):
        tsuchi.add()
    assert len(call(list_conversations, "alpha")) == 205
    offsets = [
        params["offset"]
        for method, _, params, _ in tsuchi.requests
        if method == "GET" and params.get("kinds") == "conversation"
    ]
    assert offsets == ["0", "200"]


def test_create_chat_outside_and_inside_a_space(tsuchi, call):
    created = call(create_conversation, "alpha", ConversationCreate())
    assert (created.kind, created.title, created.parent_id) == (
        "conversation",
        "New chat",
        None,
    )
    space = tsuchi.add("project", "Garden")
    created = call(
        create_conversation,
        "alpha",
        ConversationCreate(title=" Beds ", space_id=space["id"]),
    )
    assert (created.title, created.parent_id) == ("Beds", UUID(space["id"]))


@pytest.mark.parametrize(
    "parent",
    [
        None,
        {"kind": "project", "deleted_at": "2026-11-02T00:00:00+00:00"},
        {"kind": "space"},
        {"kind": "conversation"},
    ],
)
def test_create_rejects_a_missing_or_invalid_space(tsuchi, call, parent):
    space_id = tsuchi.add(**parent)["id"] if parent else str(uuid4())
    with pytest.raises(ServiceError, match="This Space no longer exists") as error:
        call(create_conversation, "alpha", ConversationCreate(space_id=space_id))
    assert error.value.status == 404
    assert tsuchi.writes("POST") == []


@pytest.mark.parametrize(
    "body",
    [{"title": None}, {"title": " "}, {"title": "x" * 501}, {"parent_id": None}],
)
def test_update_rejects_blank_null_or_unknown_fields(body):
    with pytest.raises(ValidationError):
        update(**body)


def test_update_changes_only_fields_in_the_body(tsuchi, call):
    space = tsuchi.add("project", "Garden")
    chat = tsuchi.add(title="Chain", parent_id=space["id"], tags=["kept"])
    tsuchi.message(chat, "user", "Hello")
    chat_id = UUID(chat["id"])

    renamed = call(update_conversation, "alpha", chat_id, update(title=" Plans "))
    assert (renamed.title, renamed.space_id) == ("Plans", UUID(space["id"]))
    assert chat["tags"] == ["kept"] and len(chat["messages"]) == 1

    moved_out = call(update_conversation, "alpha", chat_id, update(space_id=None))
    assert (moved_out.title, moved_out.space_id) == ("Plans", None)
    assert chat["parent_id"] is None

    moved_in = call(update_conversation, "alpha", chat_id, update(space_id=space["id"]))
    assert moved_in.space_id == UUID(space["id"])
    assert chat["parent_id"] == space["id"]


def test_empty_update_does_not_write(tsuchi, call):
    chat = tsuchi.add()
    summary = call(update_conversation, "alpha", UUID(chat["id"]), update())
    assert summary.version == 1
    assert tsuchi.writes("PUT") == []


def test_update_retries_when_a_reply_lands_first(tsuchi, call):
    chat = tsuchi.add()
    tsuchi.message(chat, "user", "Hello")
    replies = ["Hi there"]

    def reply_arrives(resource_id):
        if replies:
            tsuchi.message(tsuchi.resources[resource_id], "assistant", replies.pop())

    tsuchi.before_write = reply_arrives
    summary = call(update_conversation, "alpha", UUID(chat["id"]), update(title="Hi"))
    assert summary.title == "Hi"
    assert [item["content"] for item in chat["messages"]] == ["Hello", "Hi there"]
    assert len(tsuchi.writes("PUT")) == 2


def test_update_reports_a_busy_chat_after_repeated_conflicts(tsuchi, call):
    chat = tsuchi.add()
    tsuchi.before_write = lambda resource_id: tsuchi.message(
        tsuchi.resources[resource_id], "assistant", "Still talking"
    )
    with pytest.raises(ServiceError, match="busy") as error:
        call(update_conversation, "alpha", UUID(chat["id"]), update(title="Hi"))
    assert error.value.status == 409
    assert len(tsuchi.writes("PUT")) == 5


def test_update_rejects_a_missing_chat(tsuchi, call):
    space = tsuchi.add("project")
    with pytest.raises(ServiceError, match="This chat no longer exists") as error:
        call(update_conversation, "alpha", UUID(space["id"]), update(title="Hi"))
    assert error.value.status == 404


def test_delete_moves_the_chat_to_trash(tsuchi, call):
    chat = tsuchi.add()
    tsuchi.message(chat, "user", "Hello")
    response = call(delete_conversation, "alpha", UUID(chat["id"]))
    assert isinstance(response, Response) and response.status_code == 204
    assert chat["deleted_at"] is not None
    [(_, _, params, _)] = [item for item in tsuchi.requests if item[0] == "DELETE"]
    assert params == {"expected_version": "2"}
    with pytest.raises(ServiceError, match="This chat no longer exists"):
        call(delete_conversation, "alpha", UUID(chat["id"]))
