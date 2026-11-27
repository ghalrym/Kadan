import pytest
from pydantic import ValidationError

from kadan.api.routes.agents import OwnedPersonaInput


@pytest.mark.parametrize(
    "body",
    [
        {"title": " ", "markdown": "Be helpful.", "expected_version": 1},
        {"markdown": "   ", "expected_version": None},
        {"markdown": "x" * 16001, "expected_version": None},
        {"markdown": "Be helpful.", "expected_version": 0},
        {"markdown": "Be helpful.", "extra": True},
    ],
)
def test_owned_persona_input_rejects_invalid_or_blank_fields(body):
    with pytest.raises(ValidationError):
        OwnedPersonaInput.model_validate(body)


def test_owned_persona_input_allows_omitting_title():
    parsed = OwnedPersonaInput.model_validate(
        {"markdown": "Be concise.", "expected_version": None}
    )
    assert parsed.title is None
    assert parsed.markdown == "Be concise."
