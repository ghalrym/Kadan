"""Workspace persistence units; storage is a recording boundary, not Tsuchi."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from kadan.api.config import ServiceError
from kadan.api.pydantic_models.resources import Resource
from kadan.runner.tools import TOOLS_BY_NAME
from kadan.runner.workspace import (
    PROJECT_TAG,
    STATUS_PREFIX,
    AgentWorkspace,
    ProjectStatus,
    document_content,
    project_status,
)


def document(**changes):
    now = datetime.now(UTC)
    return Resource.model_validate(
        {
            "id": uuid4(),
            "agent_key": "haru",
            "kind": "space",
            "title": "Launch plan",
            "markdown": "Goal: launch.\n- [ ] Draft announcement\nUser note: keep it short.",
            "version": 3,
            "created_at": now,
            "updated_at": now,
            **changes,
        }
    )


def project(**changes):
    return document(tags=[PROJECT_TAG, STATUS_PREFIX + "active"], **changes)


class Storage:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def json_request(self, service, method, path, **kwargs):
        self.calls.append((service, method, path, kwargs))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.mark.parametrize("status", ProjectStatus)
def test_explicit_project_states_are_read_from_tags(status):
    assert (
        project_status(document(tags=[PROJECT_TAG, STATUS_PREFIX + status])) == status
    )


@pytest.mark.parametrize(
    "tags",
    [
        [PROJECT_TAG],
        [PROJECT_TAG, STATUS_PREFIX + "invented"],
        [PROJECT_TAG, STATUS_PREFIX + "active", STATUS_PREFIX + "completed"],
    ],
)
def test_invalid_user_edited_state_needs_attention_instead_of_becoming_active(tags):
    with pytest.raises(ServiceError, match="status tags changed"):
        project_status(document(tags=tags))


def test_ordinary_documents_are_not_implicitly_projects():
    assert project_status(document(tags=[STATUS_PREFIX + "active"])) is None


def test_create_project_uses_an_existing_space_in_the_running_agent_namespace():
    saved = project()
    storage = Storage(saved.model_dump(mode="json"))
    result = asyncio.run(
        AgentWorkspace(storage, "haru").create_project(saved.title, saved.markdown)
    )
    assert result["id"] == str(saved.id)
    assert result["project_status"] == "active"
    assert storage.calls == [
        (
            "Tsuchi",
            "POST",
            "/api/agents/haru/resources",
            {
                "json": {
                    "kind": "space",
                    "title": saved.title,
                    "markdown": saved.markdown,
                    "tags": [PROJECT_TAG, STATUS_PREFIX + "active"],
                }
            },
        )
    ]


def test_index_exact_filters_search_matches_and_preserves_pagination():
    saved = project()
    unrelated = document(markdown="References kadan-project but is not a project")
    storage = Storage(
        [unrelated.model_dump(mode="json"), saved.model_dump(mode="json")]
    )
    result = asyncio.run(
        AgentWorkspace(storage, "haru").list(projects_only=True, limit=2, offset=10)
    )
    assert result == {
        "documents": [
            {
                "id": str(saved.id),
                "title": saved.title,
                "parent_id": None,
                "version": 3,
                "project_status": "active",
            }
        ],
        "next_offset": 12,
    }
    params = storage.calls[0][3]["params"]
    assert params == {
        "kind": "space",
        "q": PROJECT_TAG,
        "trash": "false",
        "limit": 2,
        "offset": 10,
    }


def test_empty_filtered_search_page_can_still_have_a_next_page():
    storage = Storage([document().model_dump(mode="json")])
    result = asyncio.run(
        AgentWorkspace(storage, "haru").list(projects_only=True, limit=1)
    )
    assert result == {"documents": [], "next_offset": 1}


def test_document_discovery_keeps_the_requested_search():
    saved = document()
    storage = Storage([saved.model_dump(mode="json")])
    result = asyncio.run(AgentWorkspace(storage, "haru").list(query="launch"))
    assert result["documents"][0]["id"] == str(saved.id)
    assert result["next_offset"] is None
    assert storage.calls[0][3]["params"]["q"] == "launch"


def test_project_discovery_combines_the_marker_and_requested_search():
    storage = Storage([])
    asyncio.run(
        AgentWorkspace(storage, "haru").list(query="launch", projects_only=True)
    )
    assert storage.calls[0][3]["params"]["q"] == "kadan-project launch"


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "memory"},
        {"kind": "persona"},
        {"kind": "conversation"},
        {"agent_key": "another"},
        {"deleted_at": datetime.now(UTC)},
    ],
)
def test_read_excludes_other_agent_records_trash_and_nondocuments(changes):
    saved = document(**changes)
    storage = Storage(saved.model_dump(mode="json"))
    with pytest.raises(ServiceError):
        asyncio.run(AgentWorkspace(storage, "haru").get(saved.id))
    assert len(storage.calls) == 1


def test_known_decks_can_be_read_but_not_rewritten_as_spaces():
    saved = document(
        kind="deck",
        deck={"schema_version": 1, "width": 100, "height": 100, "slides": []},
    )
    storage = Storage(saved.model_dump(mode="json"))
    workspace = AgentWorkspace(storage, "haru")
    assert (
        document_content(asyncio.run(workspace.get(saved.id)))["deck"]["slides"] == []
    )
    with pytest.raises(ServiceError, match="requires a space"):
        asyncio.run(workspace.update(saved, saved.version, "replacement"))
    assert len(storage.calls) == 1


def test_project_checkpoint_preserves_user_content_and_unrelated_metadata():
    current = project(
        parent_id=uuid4(),
        attachment_ids=[uuid4()],
        sources=[{"label": "Brief", "url": "https://example.com"}],
    )
    current.tags.append("user-label")
    markdown = current.markdown + "\nDecision needed: morning or afternoon?"
    updated = current.model_copy(
        update={
            "markdown": markdown,
            "version": 4,
            "tags": [PROJECT_TAG, "user-label", STATUS_PREFIX + "awaiting_decision"],
        }
    )
    storage = Storage(updated.model_dump(mode="json"))
    result = asyncio.run(
        AgentWorkspace(storage, "haru").update(
            current, 3, markdown, ProjectStatus.AWAITING_DECISION
        )
    )
    body = storage.calls[0][3]["json"]
    assert body == {
        **current.model_dump(
            mode="json",
            include={
                "kind",
                "title",
                "parent_id",
                "sources",
                "attachment_ids",
                "deck",
                "messages",
            },
        ),
        "tags": updated.tags,
        "markdown": markdown,
        "expected_version": 3,
    }
    assert result["project_status"] == "awaiting_decision"
    assert "User note: keep it short." in result["markdown"]


def test_supporting_document_update_preserves_links_tags_and_attachments():
    current = document(parent_id=uuid4(), tags=["user-label"], attachment_ids=[uuid4()])
    storage = Storage(
        current.model_copy(update={"markdown": "Revised", "version": 4}).model_dump(
            mode="json"
        )
    )
    result = asyncio.run(AgentWorkspace(storage, "haru").update(current, 3, "Revised"))
    assert result["project_status"] is None
    body = storage.calls[0][3]["json"]
    assert body["parent_id"] == str(current.parent_id)
    assert body["attachment_ids"] == list(map(str, current.attachment_ids))
    assert body["tags"] == ["user-label"]


def test_stale_plan_does_not_write_over_a_user_edit():
    storage = Storage(None)
    with pytest.raises(ServiceError, match="changed in Tsuchi"):
        asyncio.run(
            AgentWorkspace(storage, "haru").update(
                project(version=4), 3, "Stale", ProjectStatus.ACTIVE
            )
        )
    assert storage.calls == []


@pytest.mark.parametrize(
    ("current", "status"), [(project(), None), (document(), ProjectStatus.ACTIVE)]
)
def test_plan_state_cannot_be_bypassed_through_ordinary_document_updates(
    current, status
):
    storage = Storage(None)
    with pytest.raises(ServiceError, match="Use update_project"):
        asyncio.run(
            AgentWorkspace(storage, "haru").update(current, 3, "Replacement", status)
        )
    assert storage.calls == []


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("create_project", {"title": " ", "markdown": "Plan"}),
        ("create_project", {"title": "Plan", "markdown": ""}),
        ("create_project", {"title": "Plan", "markdown": "Plan", "agent_key": "other"}),
        ("read_workspace_document", {"document_id": "../other"}),
        ("list_workspace_documents", {"limit": 51}),
        (
            "update_project",
            {
                "document_id": str(uuid4()),
                "expected_version": 1,
                "markdown": "Plan",
                "status": "running",
            },
        ),
        (
            "update_project",
            {"document_id": str(uuid4()), "expected_version": 1, "markdown": "Plan"},
        ),
        (
            "update_workspace_document",
            {"document_id": str(uuid4()), "expected_version": 0, "markdown": "Plan"},
        ),
    ],
)
def test_project_tool_schemas_reject_invalid_or_unscoped_arguments(tool, arguments):
    with pytest.raises(ValidationError):
        TOOLS_BY_NAME[tool].args_schema.model_validate(arguments)
