"""History views use only stored delivered chat and bounded content."""

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from kadan.api.config import ServiceError
from kadan.api.pydantic_models.agent_runs import RunMessage
from kadan.api.pydantic_models.resources import Message, Resource
from kadan.runner.agent_runs import RunConversation, transcript_messages
from kadan.runner.conversation_history import (
    ReadHistoryArguments,
    SearchHistoryArguments,
    history_page,
)
from kadan.runner.tools.history import read_conversation_history


@pytest.fixture
def conversation():
    now = datetime.now(UTC)
    return Resource(
        id=uuid4(),
        agent_key="test",
        kind="conversation",
        title="Chat",
        version=7,
        created_at=now,
        updated_at=now,
        messages=[
            Message(role="user", content="Older NEEDLE evidence"),
            *[
                Message(role="assistant", content=f"Reply {index}")
                for index in range(120)
            ],
        ],
    )


def test_search_reaches_message_outside_recent_97(conversation):
    result = history_page(conversation, SearchHistoryArguments(query="needle"))
    assert result["messages"][0]["id"] == str(conversation.messages[0].id)
    assert (
        result["messages"][0]["created_at"]
        == conversation.messages[0].created_at.isoformat()
    )
    assert result["messages"][0]["role"] == "user"
    assert result["whole_history_scanned"] is True
    assert result["source_version"] == 7


def test_only_delivered_chat_roles_are_eligible(conversation):
    conversation.messages = [
        Message(role="system", content="hidden"),
        Message(role="tool", content="hidden"),
        *conversation.messages,
    ]
    result = history_page(conversation, ReadHistoryArguments(limit=1))
    assert result["eligible_messages"] == 121
    assert result["messages"][0]["role"] == "user"
    with pytest.raises(ValueError, match="not delivered chat"):
        history_page(
            conversation,
            ReadHistoryArguments(
                start_message_id=conversation.messages[0].id, expected_version=7
            ),
        )


def test_search_scan_limit_reports_coverage_and_continuation(conversation):
    result = history_page(
        conversation, SearchHistoryArguments(query="absent", max_scan=3)
    )
    assert result["messages"] == []
    assert result["scanned_eligible_messages"] == 3
    assert result["scanned_last_message_id"] == str(conversation.messages[2].id)
    assert result["more_available"] is True
    assert result["whole_history_scanned"] is False
    following = history_page(
        conversation,
        SearchHistoryArguments(
            query="absent",
            max_scan=3,
            start_message_id=result["next_message_id"],
            expected_version=7,
        ),
    )
    assert following["start_eligible_offset"] == 3


def test_read_character_cut_can_be_resumed_without_losing_text(conversation):
    first = history_page(conversation, ReadHistoryArguments(max_characters=5))
    second = history_page(
        conversation,
        ReadHistoryArguments(
            start_message_id=first["next_message_id"],
            expected_version=7,
            content_offset=first["next_content_offset"],
            limit=1,
        ),
    )
    assert first["content_cut"] is True
    assert first["next_content_offset"] == 5
    assert (
        first["messages"][0]["content"] + second["messages"][0]["content"]
        == conversation.messages[0].content
    )
    assert second["next_message_id"] == str(conversation.messages[1].id)


def test_search_match_and_excerpt_limits(conversation):
    result = history_page(
        conversation,
        SearchHistoryArguments(
            query="reply", limit=2, excerpt_characters=3, max_characters=5
        ),
    )
    assert len(result["messages"]) == 2
    assert result["returned_content_characters"] == 5
    assert result["content_cut"] is True
    assert result["scanned_eligible_messages"] == 3
    assert result["next_message_id"] == str(conversation.messages[3].id)


@pytest.mark.parametrize(
    "arguments",
    [
        {"limit": 0},
        {"limit": 21},
        {"max_characters": 16001},
        {"content_offset": -1},
        {"conversation_id": str(uuid4())},
    ],
)
def test_read_bounds_and_scope_cannot_be_overridden(arguments):
    with pytest.raises(ValidationError):
        ReadHistoryArguments(**arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        {"max_scan": 501, "query": "x"},
        {"query": " "},
        {"query": "x", "excerpt_characters": 1001},
    ],
)
def test_search_bounds(arguments):
    with pytest.raises(ValidationError):
        SearchHistoryArguments(**arguments)


@pytest.mark.parametrize(
    "arguments, error",
    [
        ({"expected_version": 6}, "changed"),
        ({"start_message_id": uuid4()}, "requires expected_version"),
        ({"content_offset": 1}, "requires a message anchor"),
    ],
)
def test_invalid_pagination_is_visible(conversation, arguments, error):
    with pytest.raises(ValueError, match=error):
        history_page(conversation, ReadHistoryArguments(**arguments))


def history_binding(conversation):
    run = SimpleNamespace(
        id=uuid4(), agent_key="test", conversation_id=conversation.id, transcript=[]
    )
    runs = SimpleNamespace(
        lock=asyncio.Lock(),
        owned=AsyncMock(return_value=run),
        services=SimpleNamespace(resource=AsyncMock(return_value=conversation)),
    )
    return RunConversation(runs, run, replayed=True)


def test_tool_is_read_only_scoped_and_replayable(conversation):
    binding = history_binding(conversation)
    result = asyncio.run(
        read_conversation_history.ainvoke(
            {}, config={"configurable": {"conversation": binding}}
        )
    )
    binding.runs.services.resource.assert_awaited_once_with("test", conversation.id)
    assert result["messages"]
    assert not binding.end_run
    messages = transcript_messages(
        [
            RunMessage(
                role="assistant",
                content='{"tool":"read_conversation_history","arguments":{}}',
            ),
            RunMessage(role="user", content="Tool result: " + json.dumps(result)),
        ]
    )
    assert isinstance(messages[-1], ToolMessage)
    assert json.loads(messages[-1].content)["messages"][0]["role"] == "user"


@pytest.mark.parametrize(
    "change",
    ["agent_key", "id", "kind", "deleted_at", "fetch", "ownership", "run_scope"],
)
def test_unavailable_or_wrong_scope_is_an_error_result(conversation, change):
    binding = history_binding(conversation)
    if change in {"agent_key", "id", "kind", "deleted_at"}:
        setattr(
            conversation,
            change,
            {
                "agent_key": "other",
                "id": uuid4(),
                "kind": "space",
                "deleted_at": datetime.now(UTC),
            }[change],
        )
    elif change == "run_scope":
        binding.runs.owned.return_value = SimpleNamespace(
            agent_key="other", conversation_id=conversation.id
        )
    else:
        boundary = (
            binding.runs.owned
            if change == "ownership"
            else binding.runs.services.resource
        )
        boundary.side_effect = ServiceError("unavailable", 409)
    result = asyncio.run(binding.conversation_history(ReadHistoryArguments()))
    assert result["error"]
    assert result["history_available"] is False
    assert "messages" not in result


def test_empty_history_has_explicit_complete_coverage(conversation):
    conversation.messages = []
    result = history_page(conversation, SearchHistoryArguments(query="missing"))
    assert result["whole_history_scanned"] is True
    assert result["scanned_eligible_messages"] == 0
    assert result["next_message_id"] is None
    assert not result["more_available"]


def test_small_search_excerpt_starts_at_match(conversation):
    result = history_page(
        conversation, SearchHistoryArguments(query="needle", max_characters=1)
    )
    assert result["messages"][0]["content"] == "N"
    assert result["character_limit_reached"] is True


def test_foreign_and_out_of_range_anchors_fail(conversation):
    with pytest.raises(ValueError, match="not delivered chat"):
        history_page(
            conversation,
            ReadHistoryArguments(start_message_id=uuid4(), expected_version=7),
        )
    with pytest.raises(ValueError, match="exceeds"):
        history_page(
            conversation,
            ReadHistoryArguments(
                start_message_id=conversation.messages[0].id,
                expected_version=7,
                content_offset=10000,
            ),
        )


def test_read_continuation_does_not_claim_whole_history_coverage(conversation):
    conversation.messages = conversation.messages[:1]
    result = history_page(
        conversation,
        ReadHistoryArguments(
            start_message_id=conversation.messages[0].id,
            expected_version=7,
            content_offset=5,
        ),
    )
    assert result["whole_history_scanned"] is False
    assert result["more_available"] is False
    assert result["content_cut"] is True
