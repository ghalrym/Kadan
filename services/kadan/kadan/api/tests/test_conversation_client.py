from datetime import UTC, datetime
from uuid import uuid4

import pytest

from kadan.api.conversation_client import summarize
from kadan.api.pydantic_models.resources import Message, Resource

NOW = datetime(2026, 11, 1, tzinfo=UTC)


def chat(title, *messages):
    return Resource(
        id=uuid4(),
        agent_key="alpha",
        kind="conversation",
        title=title,
        messages=[Message(role=role, content=content) for role, content in messages],
        version=1,
        created_at=NOW,
        updated_at=NOW,
    )


@pytest.mark.parametrize(
    ("title", "messages", "shown"),
    [
        ("Chain", [("user", "Hello\n  there")], "Hello there"),
        (
            " New chat ",
            [("assistant", "Hi"), ("user", " "), ("user", "Bulbs")],
            "Bulbs",
        ),
        ("New chat", [("assistant", "Hi")], "New chat"),
        ("Chain", [], "Chain"),
        ("Bulbs", [("user", "Something else")], "Bulbs"),
        ("New chat", [("user", "x" * 61)], "x" * 59 + "…"),
        ("New chat", [("user", "x" * 60)], "x" * 60),
    ],
)
def test_default_titles_show_the_first_user_message(title, messages, shown):
    assert summarize(chat(title, *messages)).title == shown
