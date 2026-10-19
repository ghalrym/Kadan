import pytest
from pydantic import ValidationError

from kadan.api.routes.agents.conversations.runs import RunInput


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
def test_run_requests_reject_unsupported_or_ambiguous_input(body):
    with pytest.raises(ValidationError):
        RunInput.model_validate(body)
