import pytest
from pydantic import ValidationError

from kadan.api.routes.agents import AgentInput


def test_agent_key_cannot_escape_its_namespace():
    with pytest.raises(ValidationError):
        AgentInput(key="../../another-agent", name="Name")
