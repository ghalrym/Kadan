import pytest
from pydantic import ValidationError

from kadan.api.routes.agents.conversations.chat import ChatInput


@pytest.mark.parametrize(
    "body",
    [
        {"content": "Hi", "expected_version": 1},
        {"content": " ", "expected_version": 1, "persona_id": None},
        {
            "content": "Hi",
            "expected_version": 1,
            "persona_id": None,
            "tools": [{"type": "shell"}],
        },
    ],
)
def test_chat_requests_reject_unsupported_or_ambiguous_input(body):
    with pytest.raises(ValidationError):
        ChatInput.model_validate(body)
