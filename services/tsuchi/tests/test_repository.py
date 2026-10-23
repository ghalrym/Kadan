from contextlib import nullcontext
from unittest.mock import Mock
from uuid import uuid4

import pytest
from tsuchi.config import StorageError
from tsuchi.models import (
    AgentInput,
    Asset,
    ConversationContent,
    ConversationResource,
    Deck,
    DeckContent,
    ExportManifest,
    MarkdownContent,
    Message,
    ResourceKind,
    ResourceRecord,
    Slide,
    resource_input_adapter,
    utc_now,
)
from tsuchi.repository import Repository, search_expression


@pytest.fixture
def repository(tmp_path):
    repository = Repository(tmp_path)
    repository.initialize()
    repository.create_agent(AgentInput(key="alpha", name="Alpha"))
    repository.create_agent(AgentInput(key="beta", name="Beta"))
    return repository


def content(resource, **changes):
    return resource_input_adapter.validate_python(
        {**resource.model_dump(exclude=set(ResourceRecord.model_fields)), **changes}
    )


def create_asset(repository, key="alpha", is_image=False):
    now = utc_now()
    asset = Asset(
        id=uuid4(),
        agent_key=key,
        title="File",
        filename="image.png" if is_image else "file.txt",
        media_type="image/png" if is_image else "text/plain",
        size_bytes=3,
        sha256="abc",
        is_image=is_image,
        version=1,
        created_at=now,
        updated_at=now,
    )
    return repository.create_asset(asset)


def test_namespace_lookup_cannot_read_another_agents_resource(repository):
    memory = repository.create_resource(
        "alpha",
        MarkdownContent(kind="memory", title="Private alpha", markdown="Remember"),
    )
    with pytest.raises(StorageError) as error:
        repository.get_resource("beta", memory.id)
    assert error.value.status_code == 404
    assert repository.list_resources("beta", query="Remember") == []


def test_namespace_links_cannot_reference_another_agents_assets(repository):
    asset = create_asset(repository, key="beta")
    with pytest.raises(StorageError, match="Asset not found"):
        repository.create_resource(
            "alpha",
            MarkdownContent(kind="memory", title="Invalid", attachment_ids=[asset.id]),
        )
    assert repository.list_resources("alpha") == []


def test_namespace_links_cannot_reference_another_agents_resources(repository):
    resource = repository.create_resource(
        "beta", MarkdownContent(kind="memory", title="Other agent")
    )
    message = Message(
        id=uuid4(),
        role="assistant",
        content="Linked",
        created_at=utc_now(),
        resource_ids=[resource.id],
    )
    with pytest.raises(StorageError, match="Resource not found"):
        repository.create_resource(
            "alpha",
            ConversationContent(
                kind="conversation", title="Invalid", messages=[message]
            ),
        )


def test_stale_update_keeps_current_content_and_history(repository):
    memory = repository.create_resource(
        "alpha", MarkdownContent(kind="memory", title="Original")
    )
    repository.update_resource(
        "alpha", memory.id, content(memory, title="Changed"), memory.version
    )
    with pytest.raises(StorageError) as error:
        repository.update_resource(
            "alpha", memory.id, content(memory, title="Lost update"), memory.version
        )
    assert error.value.status_code == 409
    assert repository.get_resource("alpha", memory.id).title == "Changed"
    assert len(repository.versions("alpha", memory.id)) == 2


def test_history_restore_creates_a_new_revision_and_updates_search(repository):
    memory = repository.create_resource(
        "alpha", MarkdownContent(kind="memory", title="First", markdown="Moonstone")
    )
    updated = repository.update_resource(
        "alpha", memory.id, content(memory, markdown="Sunrise"), memory.version
    )
    restored = repository.restore_version("alpha", memory.id, 1, updated.version)
    assert (restored.markdown, restored.version) == ("Moonstone", 3)
    assert [version.version for version in repository.versions("alpha", memory.id)] == [
        3,
        2,
        1,
    ]
    assert repository.list_resources("alpha", query="Sunrise") == []
    assert [
        item.id for item in repository.list_resources("alpha", query="Moonstone")
    ] == [memory.id]


def test_search_operators_are_literal_tokens():
    assert (
        search_expression('"hello" OR title:world* -exclude')
        == '"hello"* AND "OR"* AND "title"* AND "world"* AND "exclude"*'
    )
    assert search_expression('"**()') == ""


def test_fts_search_matches_structured_slide_text_and_tags(repository):
    deck = repository.create_resource(
        "alpha",
        DeckContent(
            kind="deck",
            title="Plan",
            tags=["release"],
            deck=Deck(slides=[Slide(id=uuid4(), layout="text", notes="Milestone")]),
        ),
    )
    assert [
        item.id
        for item in repository.list_resources("alpha", query="release milestone")
    ] == [deck.id]
    assert repository.list_resources("alpha", query='"***()') == []


def test_search_with_30_words_requires_every_word(repository):
    words = [f"term{index}" for index in range(30)]
    complete = repository.create_resource(
        "alpha",
        MarkdownContent(kind="memory", title="Complete", markdown=" ".join(words)),
    )
    repository.create_resource(
        "alpha",
        MarkdownContent(
            kind="memory", title="Missing last word", markdown=" ".join(words[:-1])
        ),
    )
    assert [
        item.id for item in repository.list_resources("alpha", query=" ".join(words))
    ] == [complete.id]


def test_search_with_31_words_reports_an_error_instead_of_partial_matches(repository):
    words = [f"term{index}" for index in range(30)]
    repository.create_resource(
        "alpha",
        MarkdownContent(kind="memory", title="Partial", markdown=" ".join(words)),
    )
    with pytest.raises(StorageError, match="30 words") as error:
        repository.list_resources("alpha", query=" ".join([*words, "absent"]))
    assert error.value.status_code == 422


def test_space_cycle_is_rejected(repository):
    parent = repository.create_resource(
        "alpha", MarkdownContent(kind="space", title="Parent")
    )
    child = repository.create_resource(
        "alpha", MarkdownContent(kind="space", title="Child", parent_id=parent.id)
    )
    with pytest.raises(StorageError, match="ancestors"):
        repository.update_resource(
            "alpha", parent.id, content(parent, parent_id=child.id), parent.version
        )
    assert repository.get_resource("alpha", parent.id).parent_id is None


def test_space_with_active_child_cannot_be_trashed(repository):
    parent = repository.create_resource(
        "alpha", MarkdownContent(kind="space", title="Parent")
    )
    repository.create_resource(
        "alpha", MarkdownContent(kind="memory", title="Child", parent_id=parent.id)
    )
    with pytest.raises(StorageError, match="active children"):
        repository.trash_resource("alpha", parent.id, parent.version)


def test_resource_restore_requires_active_parent(repository):
    parent = repository.create_resource(
        "alpha", MarkdownContent(kind="space", title="Parent")
    )
    child = repository.create_resource(
        "alpha", MarkdownContent(kind="memory", title="Child", parent_id=parent.id)
    )
    child = repository.trash_resource("alpha", child.id, child.version)
    parent = repository.trash_resource("alpha", parent.id, parent.version)
    with pytest.raises(StorageError, match="active Space"):
        repository.restore_resource("alpha", child.id, child.version)
    repository.restore_resource("alpha", parent.id, parent.version)
    assert (
        repository.restore_resource("alpha", child.id, child.version).deleted_at is None
    )


def test_non_image_asset_cannot_become_slide_image(repository):
    asset = create_asset(repository)
    with pytest.raises(StorageError, match="verified raster image"):
        repository.create_resource(
            "alpha",
            DeckContent(
                kind="deck",
                title="Invalid",
                deck=Deck(
                    slides=[Slide(id=uuid4(), layout="image", image_asset_id=asset.id)]
                ),
            ),
        )


@pytest.mark.parametrize("location", ["resource", "message", "slide"])
def test_active_asset_references_prevent_trashing(repository, location):
    asset = create_asset(repository, is_image=True)
    if location == "resource":
        value = MarkdownContent(
            kind="memory", title="Reference", attachment_ids=[asset.id]
        )
    elif location == "message":
        value = ConversationContent(
            kind="conversation",
            title="Reference",
            messages=[
                Message(
                    id=uuid4(),
                    role="user",
                    content="Attachment",
                    created_at=utc_now(),
                    attachment_ids=[asset.id],
                )
            ],
        )
    else:
        value = DeckContent(
            kind="deck",
            title="Reference",
            deck=Deck(
                slides=[Slide(id=uuid4(), layout="image", image_asset_id=asset.id)]
            ),
        )
    resource = repository.create_resource("alpha", value)
    with pytest.raises(StorageError, match="active resources"):
        repository.trash_asset("alpha", asset.id, asset.version)
    repository.trash_resource("alpha", resource.id, resource.version)
    assert (
        repository.trash_asset("alpha", asset.id, asset.version).deleted_at is not None
    )


def test_append_message_is_unique_and_versioned(repository):
    conversation = repository.create_resource(
        "alpha", ConversationContent(kind="conversation", title="Chat")
    )
    message = Message(
        id=uuid4(), role="assistant", content="Response", created_at=utc_now()
    )
    updated = repository.append_message(
        "alpha", conversation.id, message, conversation.version
    )
    assert updated.messages == [message]
    assert updated.version == 2
    with pytest.raises(StorageError, match="already exists"):
        repository.append_message("alpha", conversation.id, message, updated.version)


def test_append_messages_beyond_former_capacity_preserves_history():
    now = utc_now()
    history = [
        Message(id=uuid4(), role="user", content=str(index), created_at=now)
        for index in range(9999)
    ]
    conversation = ConversationResource(
        kind="conversation",
        title="Chain",
        id=uuid4(),
        agent_key="alpha",
        version=1,
        created_at=now,
        updated_at=now,
        messages=history,
    )
    repository = Mock(spec=Repository)
    repository.connect.side_effect = lambda **kwargs: nullcontext(None)
    repository._resource.side_effect = lambda *args: conversation

    def update(db, key, current, value, expected_version, action):
        assert expected_version == current.version
        return ConversationResource(
            **value.model_dump(),
            id=current.id,
            agent_key=key,
            version=current.version + 1,
            created_at=now,
            updated_at=now,
        )

    repository._update_resource.side_effect = update
    for role in ("user", "assistant", "user"):
        message = Message(id=uuid4(), role=role, content="Next", created_at=now)
        conversation = Repository.append_message(
            repository, "alpha", conversation.id, message, conversation.version
        )
        assert conversation.messages[-1] == message
    assert len(conversation.messages) == 10002
    assert conversation.messages[:9999] == history
    assert conversation.version == 4


def test_export_manifest_includes_trash_and_revisions_only_for_selected_agent(
    repository,
):
    memory = repository.create_resource(
        "alpha", MarkdownContent(kind="memory", title="Retained")
    )
    repository.trash_resource("alpha", memory.id, memory.version)
    asset = create_asset(repository)
    repository.trash_asset("alpha", asset.id, asset.version)
    repository.create_resource(
        "beta", MarkdownContent(kind="memory", title="Other namespace")
    )
    with repository.connect() as db:
        manifest = repository.export_manifest(db, "alpha")
    assert ExportManifest.model_validate_json(manifest.model_dump_json()) == manifest
    assert manifest.agent.key == "alpha"
    assert [item.title for item in manifest.resources] == ["Retained"]
    assert manifest.resources[0].deleted_at is not None
    assert len(manifest.versions) == 2
    assert manifest.assets[0].deleted_at is not None
    assert all(item.title != "Other namespace" for item in manifest.activity)


def test_stats_count_retained_bytes_and_separate_trash(repository):
    memory = repository.create_resource(
        "alpha", MarkdownContent(kind="memory", title="Retained")
    )
    repository.trash_resource("alpha", memory.id, memory.version)
    asset = create_asset(repository, is_image=True)
    repository.trash_asset("alpha", asset.id, asset.version)
    stats = repository.stats("alpha")
    assert stats.resources["memory"] == stats.assets == stats.images == 0
    assert stats.trashed_resources == stats.trashed_assets == 1
    assert stats.storage_bytes == 3
    assert stats.revisions == 2


def test_document_kinds_filter_before_pagination_and_keep_namespace(repository):
    root = repository.create_resource(
        "alpha", MarkdownContent(kind="space", title="Match root")
    )
    child = repository.create_resource(
        "alpha", MarkdownContent(kind="space", title="Match child", parent_id=root.id)
    )
    deck = repository.create_resource(
        "alpha", DeckContent(kind="deck", title="Match deck", deck=Deck())
    )
    memory = repository.create_resource(
        "alpha", MarkdownContent(kind="memory", title="Match memory", parent_id=root.id)
    )
    repository.create_resource(
        "alpha", MarkdownContent(kind="persona", title="Match persona")
    )
    repository.create_resource(
        "beta", MarkdownContent(kind="space", title="Match other agent")
    )
    trashed = repository.create_resource(
        "alpha", MarkdownContent(kind="space", title="Match trashed")
    )
    repository.trash_resource("alpha", trashed.id, trashed.version)
    kinds = [ResourceKind.SPACE, ResourceKind.DECK]
    documents = repository.list_resources("alpha", query="Match", kinds=kinds)
    assert {item.id for item in documents} == {root.id, child.id, deck.id}
    pages = [
        repository.list_resources(
            "alpha", query="Match", kinds=kinds, limit=1, offset=offset
        )[0]
        for offset in range(3)
    ]
    assert pages == documents
    assert repository.list_resources("alpha", query="memory", kinds=kinds) == []
    assert repository.list_resources("alpha", offset=3, kinds=kinds) == []
    assert repository.get_resource("alpha", memory.id) == memory
    assert repository.versions("alpha", memory.id)[0].snapshot == memory
    assert repository.get_resource("alpha", child.id).parent_id == root.id


def test_kind_filters_intersect_and_empty_kind_selection_is_empty(repository):
    saved = repository.create_resource(
        "alpha", MarkdownContent(kind="memory", title="Remember")
    )
    assert repository.list_resources("alpha", kinds=[]) == []
    assert (
        repository.list_resources(
            "alpha", kind=ResourceKind.MEMORY, kinds=[ResourceKind.SPACE]
        )
        == []
    )
    assert repository.list_resources(
        "alpha", kind=ResourceKind.MEMORY, kinds=[ResourceKind.MEMORY]
    ) == [saved]
