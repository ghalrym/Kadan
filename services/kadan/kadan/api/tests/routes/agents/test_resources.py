import pytest
from pydantic import ValidationError

from kadan.api.routes.agents.resources import PersonaUpdate, ResourceCreate


@pytest.mark.parametrize(
    ("model", "body"),
    [
        (ResourceCreate, {"kind": "memory", "title": "Unsupported write"}),
        (ResourceCreate, {"kind": "conversation", "title": "Second thread"}),
        (PersonaUpdate, {"title": " ", "markdown": "x", "expected_version": 1}),
    ],
)
def test_resource_requests_reject_unsupported_or_blank_input(model, body):
    with pytest.raises(ValidationError):
        model.model_validate(body)
