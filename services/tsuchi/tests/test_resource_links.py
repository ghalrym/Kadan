"""Parent invariants with mocked storage; no database or service is started."""

from types import SimpleNamespace
from unittest.mock import Mock, call
from uuid import uuid4

import pytest
from tsuchi.config import StorageError
from tsuchi.models import (
    ConversationContent,
    MarkdownContent,
    ResourceKind,
    utc_now,
)
from tsuchi.repository import Repository


def space(parent_id=None):
    return SimpleNamespace(
        kind=ResourceKind.SPACE, deleted_at=None, parent_id=parent_id
    )


def test_parent_ancestors_are_read_in_the_child_agent_namespace():
    parent_id, ancestor_id = uuid4(), uuid4()
    repository = Repository.__new__(Repository)
    repository._resource = Mock(side_effect=[space(ancestor_id), space()])
    db = Mock()
    repository._validate_links(
        db,
        "alpha",
        MarkdownContent(kind="space", title="Child", parent_id=parent_id),
        uuid4(),
    )
    assert repository._resource.call_args_list == [
        call(db, "alpha", parent_id),
        call(db, "alpha", ancestor_id),
    ]
    db.execute.assert_not_called()


@pytest.mark.parametrize("kind", ["memory", "persona", "deck", "conversation"])
def test_only_spaces_can_be_document_parents(kind):
    repository = Repository.__new__(Repository)
    repository._resource = Mock(
        return_value=SimpleNamespace(kind=kind, deleted_at=None)
    )
    with pytest.raises(StorageError, match="active Space"):
        repository._validate_links(
            Mock(),
            "alpha",
            MarkdownContent(kind="space", title="Child", parent_id=uuid4()),
            uuid4(),
        )


def test_trashed_parent_is_rejected():
    repository = Repository.__new__(Repository)
    repository._resource = Mock(
        return_value=SimpleNamespace(kind=ResourceKind.SPACE, deleted_at=utc_now())
    )
    with pytest.raises(StorageError, match="active Space"):
        repository._validate_links(
            Mock(),
            "alpha",
            MarkdownContent(kind="space", title="Child", parent_id=uuid4()),
            uuid4(),
        )


def test_parent_missing_from_the_agent_namespace_is_rejected():
    parent_id = uuid4()
    repository = Repository.__new__(Repository)
    repository._resource = Mock(side_effect=StorageError("Resource not found", 404))
    db = Mock()
    with pytest.raises(StorageError) as error:
        repository._validate_links(
            db,
            "alpha",
            MarkdownContent(kind="space", title="Child", parent_id=parent_id),
            uuid4(),
        )
    assert error.value.status_code == 404
    repository._resource.assert_called_once_with(db, "alpha", parent_id)
    db.execute.assert_not_called()


@pytest.mark.parametrize("self_parent", [True, False])
def test_self_parent_and_cyclic_ancestors_are_rejected(self_parent):
    child_id, parent_id, ancestor_id = uuid4(), uuid4(), uuid4()
    repository = Repository.__new__(Repository)
    repository._resource = Mock(side_effect=[space(ancestor_id), space(parent_id)])
    with pytest.raises(StorageError, match="ancestors"):
        repository._validate_links(
            Mock(),
            "alpha",
            MarkdownContent(
                kind="space",
                title="Child",
                parent_id=child_id if self_parent else parent_id,
            ),
            child_id,
        )
    assert repository._resource.call_count == (0 if self_parent else 2)


def project(deleted_at=None):
    return SimpleNamespace(
        kind=ResourceKind.PROJECT, deleted_at=deleted_at, parent_id=None
    )


def test_project_cannot_have_a_parent():
    repository = Repository.__new__(Repository)
    repository._resource = Mock(return_value=space())
    with pytest.raises(StorageError, match="project cannot be placed"):
        repository._validate_links(
            Mock(),
            "alpha",
            MarkdownContent(kind="project", title="Trip", parent_id=uuid4()),
            uuid4(),
        )
    repository._resource.assert_not_called()


def test_conversation_can_be_placed_in_a_project():
    parent_id = uuid4()
    repository = Repository.__new__(Repository)
    repository._resource = Mock(return_value=project())
    db = Mock()
    repository._validate_links(
        db,
        "alpha",
        ConversationContent(kind="conversation", title="Chat", parent_id=parent_id),
        uuid4(),
    )
    repository._resource.assert_called_once_with(db, "alpha", parent_id)


@pytest.mark.parametrize("kind", ["space", "memory", "persona"])
def test_only_conversations_can_be_placed_in_a_project(kind):
    repository = Repository.__new__(Repository)
    repository._resource = Mock(return_value=project())
    with pytest.raises(StorageError, match="Only conversations"):
        repository._validate_links(
            Mock(),
            "alpha",
            MarkdownContent(kind=kind, title="Child", parent_id=uuid4()),
            uuid4(),
        )


def test_project_cannot_be_an_ancestor_of_a_nested_space():
    repository = Repository.__new__(Repository)
    repository._resource = Mock(side_effect=[space(uuid4()), project()])
    with pytest.raises(StorageError, match="Only conversations"):
        repository._validate_links(
            Mock(),
            "alpha",
            ConversationContent(kind="conversation", title="Chat", parent_id=uuid4()),
            uuid4(),
        )


def test_conversation_cannot_be_placed_in_a_trashed_project():
    repository = Repository.__new__(Repository)
    repository._resource = Mock(return_value=project(utc_now()))
    with pytest.raises(StorageError, match="active project"):
        repository._validate_links(
            Mock(),
            "alpha",
            ConversationContent(kind="conversation", title="Chat", parent_id=uuid4()),
            uuid4(),
        )
