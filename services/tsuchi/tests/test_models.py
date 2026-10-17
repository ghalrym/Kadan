from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from tsuchi.models import (
    AgentInput,
    ConversationContent,
    Deck,
    DeckContent,
    ExportRevision,
    MarkdownContent,
    MarkdownResource,
    Message,
    Slide,
    SourceLink,
    resource_input_adapter,
    utc_now,
)


@pytest.mark.parametrize("key", ["../outside", "with space", "CAPS", "", "a" * 65])
def test_agent_key_is_a_bounded_namespace_slug(key):
    with pytest.raises(ValidationError):
        AgentInput(key=key, name="Agent")


def test_resource_shape_preserves_editable_deck_data():
    slide = Slide(id=uuid4(), layout="split", title="Title", body="Text", notes="Notes")
    resource = DeckContent(kind="deck", title="Deck", deck=Deck(slides=[slide]))
    restored = resource_input_adapter.validate_json(resource.model_dump_json())
    assert restored.deck.slides == [slide]


@pytest.mark.parametrize(
    "value",
    [
        {"kind": "deck", "title": "Missing deck"},
        {"kind": "memory", "title": "Extra deck", "deck": {"slides": []}},
    ],
)
def test_resource_kind_controls_deck_payload(value):
    with pytest.raises(ValidationError):
        resource_input_adapter.validate_python(value)


@pytest.mark.parametrize("kind", ["persona", "memory", "space"])
def test_markdown_content_keeps_null_deck_and_empty_messages(kind):
    content = resource_input_adapter.validate_python(
        {"kind": kind, "title": "Markdown", "deck": None, "messages": []}
    )
    assert isinstance(content, MarkdownContent)
    assert content.model_dump(mode="json")["deck"] is None
    assert content.model_dump(mode="json")["messages"] == []


@pytest.mark.parametrize("kind", ["persona", "memory", "space", "deck"])
def test_only_conversation_content_accepts_messages(kind):
    message = Message(id=uuid4(), role="user", content="Hello", created_at=utc_now())
    value = {"kind": kind, "title": "Content", "messages": [message]}
    if kind == "deck":
        value["deck"] = Deck()
    with pytest.raises(ValidationError):
        resource_input_adapter.validate_python(value)


def test_conversation_rejects_duplicate_message_ids():
    message = Message(
        id=uuid4(), role="user", content="Hello", created_at=datetime.now(timezone.utc)
    )
    with pytest.raises(ValidationError, match="Message IDs must be unique"):
        ConversationContent(
            kind="conversation", title="Conversation", messages=[message, message]
        )


def test_message_timestamp_requires_timezone():
    with pytest.raises(ValidationError, match="timezone"):
        Message(
            id=uuid4(), role="user", content="Hello", created_at=datetime(2026, 1, 1)
        )


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "javascript:alert(1)", "/local/path"]
)
def test_sources_require_http_urls(url):
    with pytest.raises(ValidationError):
        SourceLink(label="Source", url=url)


def test_export_revision_must_match_its_snapshot():
    now = utc_now()
    snapshot = MarkdownResource(
        kind="memory",
        title="Memory",
        id=uuid4(),
        agent_key="alpha",
        version=1,
        created_at=now,
        updated_at=now,
    )
    with pytest.raises(
        ValidationError, match="identify its resource snapshot and version"
    ):
        ExportRevision(
            resource_id=snapshot.id, version=2, created_at=now, snapshot=snapshot
        )
