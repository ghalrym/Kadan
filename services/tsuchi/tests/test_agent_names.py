from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest
from pydantic import ValidationError
from tsuchi.config import StorageError
from tsuchi.models import Agent, AgentRename
from tsuchi.repository import Repository
from tsuchi.routes import create_router


def test_rename_normalizes_name_but_preserves_expected_name_exactly():
    value = AgentRename(name="  Ember  ", expected_name=" Old name ")
    assert value.name == "Ember"
    assert value.expected_name == " Old name "


@pytest.mark.parametrize(
    "value",
    [
        {"name": " ", "expected_name": "Old"},
        {"name": "x" * 501, "expected_name": "Old"},
        {"name": "New"},
        {"name": "New", "expected_name": "Old", "key": "changed"},
    ],
)
def test_rename_rejects_invalid_input(value):
    with pytest.raises(ValidationError):
        AgentRename(**value)


@pytest.fixture
def repository():
    repository = Repository(Path("unused"))
    repository.connect = Mock(return_value=nullcontext(Mock()))
    repository._agent = Mock(
        return_value=Agent(
            key="ember",
            name="Old",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            selected_pet_id=uuid4(),
        )
    )
    repository._activity = Mock()
    return repository


def test_rename_changes_only_name_and_records_activity_in_write_transaction(repository):
    original = repository._agent.return_value
    result = repository.rename_agent(
        "ember", AgentRename(name="New", expected_name="Old")
    )
    db = repository.connect.return_value.enter_result
    repository.connect.assert_called_once_with(write=True)
    repository._agent.assert_called_once_with(db, "ember")
    db.execute.assert_called_once_with(
        "UPDATE agents SET name = ? WHERE key = ?", ("New", "ember")
    )
    repository._activity.assert_called_once_with(
        db, "ember", "updated", "agent", "ember", "New"
    )
    assert result == original.model_copy(update={"name": "New"})


def test_stale_rename_does_not_write(repository):
    with pytest.raises(StorageError) as raised:
        repository.rename_agent(
            "ember", AgentRename(name="New", expected_name="Another")
        )
    assert raised.value.status_code == 409
    repository.connect.return_value.enter_result.execute.assert_not_called()
    repository._activity.assert_not_called()


def test_retry_of_saved_name_does_not_write_again(repository):
    result = repository.rename_agent(
        "ember", AgentRename(name="Old", expected_name="Earlier")
    )
    assert result is repository._agent.return_value
    repository.connect.return_value.enter_result.execute.assert_not_called()
    repository._activity.assert_not_called()


def test_missing_agent_does_not_write(repository):
    repository._agent.side_effect = StorageError("Agent namespace not found", 404)
    with pytest.raises(StorageError) as raised:
        repository.rename_agent("missing", AgentRename(name="New", expected_name="Old"))
    assert raised.value.status_code == 404
    repository.connect.return_value.enter_result.execute.assert_not_called()


def test_routes_return_repository_agent_and_forward_expected_name():
    repository = Mock()
    router = create_router(repository, Mock())
    endpoints = {
        next(iter(route.methods)): route.endpoint
        for route in router.routes
        if route.path == "/api/agents/{key}"
    }
    value = AgentRename(name="New", expected_name="Old")
    assert endpoints["GET"]("ember") is repository.get_agent.return_value
    assert endpoints["PUT"]("ember", value) is repository.rename_agent.return_value
    repository.get_agent.assert_called_once_with("ember")
    repository.rename_agent.assert_called_once_with("ember", value)
