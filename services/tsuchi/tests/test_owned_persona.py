from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest
from pydantic import ValidationError
from tsuchi.config import StorageError
from tsuchi.models import (
    Agent,
    MarkdownResource,
    OwnedPersonaUpdate,
    SourceLink,
    utc_now,
)
from tsuchi.repository import Repository


@pytest.fixture
def storage():
    repository = Repository(Path("/unused"))
    db = Mock()
    repository.connect = Mock(return_value=nullcontext(db))
    return repository, db


def persona(**changes):
    now = utc_now()
    return MarkdownResource(
        **{
            "id": uuid4(),
            "agent_key": "alpha",
            "kind": "persona",
            "title": "Existing persona",
            "markdown": "Existing instructions",
            "created_at": now,
            "updated_at": now,
            "version": 3,
            **changes,
        }
    )


def reads(db, current=None, *, key="alpha", persona_id=None):
    agent = Agent(
        key=key,
        name="Agent",
        created_at=utc_now(),
        persona_id=current.id if current else persona_id,
    )
    db.execute.return_value.fetchone.side_effect = [
        agent.model_dump(mode="json"),
        (current.model_dump_json(),) if current else None,
    ]


def writes(db):
    return [
        call.args
        for call in db.execute.call_args_list
        if call.args[0].split()[0] in {"INSERT", "UPDATE", "DELETE"}
    ]


@pytest.mark.parametrize("has_persona_column", [False, True])
def test_initialize_adds_persona_pointer_only_when_missing(
    tmp_path, monkeypatch, has_persona_column
):
    db = Mock()
    columns = ["key", "name", "created_at", "selected_pet_id"]
    if has_persona_column:
        columns.append("persona_id")
    db.execute.side_effect = lambda sql: (
        [(index, name) for index, name in enumerate(columns)]
        if sql == "PRAGMA table_info(agents)"
        else None
    )
    monkeypatch.setattr("tsuchi.repository.sqlite3.connect", Mock(return_value=db))

    Repository(tmp_path).initialize()

    alterations = [
        call.args[0]
        for call in db.execute.call_args_list
        if call.args[0].startswith("ALTER")
    ]
    assert alterations == (
        [] if has_persona_column else ["ALTER TABLE agents ADD COLUMN persona_id TEXT"]
    )
    db.commit.assert_called_once()


@pytest.mark.parametrize(
    "changes",
    [
        {"title": " "},
        {"title": "a" * 201},
        {"markdown": "\n "},
        {"markdown": "a" * 16001},
        {"expected_version": 0},
    ],
)
def test_owned_persona_rejects_invalid_edits(changes):
    with pytest.raises(ValidationError):
        OwnedPersonaUpdate.model_validate(
            {
                "title": "Persona",
                "markdown": "Instructions",
                "expected_version": None,
                **changes,
            }
        )


def test_owned_persona_requires_explicit_expected_version():
    with pytest.raises(ValidationError):
        OwnedPersonaUpdate(title="Persona", markdown="Instructions")


def test_owned_persona_trims_title_and_preserves_markdown():
    value = OwnedPersonaUpdate(
        title="  Persona  ", markdown="  Instructions\n", expected_version=None
    )
    assert value.title == "Persona"
    assert value.markdown == "  Instructions\n"


def test_unset_persona_does_not_select_an_existing_resource(storage):
    repository, db = storage
    reads(db)

    assert repository.get_owned_persona("alpha") is None

    db.execute.assert_called_once_with("SELECT * FROM agents WHERE key = ?", ("alpha",))


def test_get_returns_the_owned_persona(storage):
    repository, db = storage
    current = persona()
    reads(db, current)

    assert repository.get_owned_persona("alpha") == current

    assert writes(db) == []


def test_first_save_assigns_owned_persona_and_records_revision_in_one_transaction(
    storage,
):
    repository, db = storage
    reads(db)

    saved = repository.save_owned_persona(
        "alpha",
        OwnedPersonaUpdate(
            title="Persona", markdown="Instructions", expected_version=None
        ),
    )

    repository.connect.assert_called_once_with(write=True)
    assert saved.version == 1
    assert saved.agent_key == "alpha"
    assert saved.kind == "persona"
    assert saved.title == "Persona"
    assert saved.markdown == "Instructions"
    db.execute.assert_any_call(
        "UPDATE agents SET persona_id = ? WHERE key = ?", (str(saved.id), "alpha")
    )
    db.execute.assert_any_call(
        "INSERT INTO revisions VALUES (?, ?, ?, ?, ?)",
        (
            "alpha",
            str(saved.id),
            1,
            saved.updated_at.isoformat(),
            saved.model_dump_json(),
        ),
    )


@pytest.mark.parametrize("expected_version", [None, 2])
def test_existing_persona_rejects_first_save_or_stale_version(
    storage, expected_version
):
    repository, db = storage
    reads(db, persona())

    with pytest.raises(StorageError) as error:
        repository.save_owned_persona(
            "alpha",
            OwnedPersonaUpdate(
                title="Persona",
                markdown="Instructions",
                expected_version=expected_version,
            ),
        )

    assert error.value.status_code == 409
    assert writes(db) == []


def test_unset_persona_rejects_an_expected_saved_version(storage):
    repository, db = storage
    reads(db)

    with pytest.raises(StorageError) as error:
        repository.save_owned_persona(
            "alpha",
            OwnedPersonaUpdate(
                title="Persona", markdown="Instructions", expected_version=1
            ),
        )

    assert error.value.status_code == 409
    assert writes(db) == []


def test_edit_preserves_identity_metadata_and_adds_a_revision(storage):
    repository, db = storage
    current = persona(
        tags=["curated"],
        sources=[SourceLink(label="Reference", url="https://example.com")],
        parent_id=uuid4(),
        attachment_ids=[uuid4()],
    )
    reads(db, current)
    repository._validate_links = Mock()

    saved = repository.save_owned_persona(
        "alpha",
        OwnedPersonaUpdate(
            title="Revised", markdown="Updated instructions", expected_version=3
        ),
    )

    assert saved.model_dump(
        exclude={"title", "markdown", "version", "updated_at"}
    ) == current.model_dump(exclude={"title", "markdown", "version", "updated_at"})
    assert saved.title == "Revised"
    assert saved.markdown == "Updated instructions"
    assert saved.version == 4
    repository.connect.assert_called_once_with(write=True)
    assert not any(sql.startswith("UPDATE agents") for sql, _ in writes(db))
    revisions = [
        params for sql, params in writes(db) if sql.startswith("INSERT INTO revisions")
    ]
    assert revisions == [
        (
            "alpha",
            str(saved.id),
            4,
            saved.updated_at.isoformat(),
            saved.model_dump_json(),
        )
    ]


@pytest.mark.parametrize("operation", ["get", "save"])
@pytest.mark.parametrize("state", ["missing", "trashed", "wrong_kind"])
def test_broken_owned_persona_is_not_replaced(storage, operation, state):
    repository, db = storage
    current = (
        None
        if state == "missing"
        else persona(
            deleted_at=utc_now() if state == "trashed" else None,
            kind="memory" if state == "wrong_kind" else "persona",
        )
    )
    reads(db, current, persona_id=uuid4())

    with pytest.raises(StorageError) as error:
        if operation == "get":
            repository.get_owned_persona("alpha")
        else:
            repository.save_owned_persona(
                "alpha",
                OwnedPersonaUpdate(
                    title="Persona", markdown="Instructions", expected_version=None
                ),
            )

    assert error.value.status_code == (404 if state == "missing" else 409)
    if state == "trashed":
        assert "Restore" in error.value.detail
    assert writes(db) == []


def test_owned_persona_lookup_stays_within_its_agent_namespace(storage):
    repository, db = storage
    target = uuid4()
    reads(db, key="beta", persona_id=target)

    with pytest.raises(StorageError) as error:
        repository.get_owned_persona("beta")

    assert error.value.status_code == 404
    db.execute.assert_any_call(
        "SELECT data FROM resources WHERE agent_key = ? AND id = ?",
        ("beta", str(target)),
    )


@pytest.mark.parametrize("operation", ["get", "save"])
def test_unknown_agent_namespace_is_rejected(storage, operation):
    repository, db = storage
    db.execute.return_value.fetchone.return_value = None

    with pytest.raises(StorageError) as error:
        if operation == "get":
            repository.get_owned_persona("missing")
        else:
            repository.save_owned_persona(
                "missing",
                OwnedPersonaUpdate(
                    title="Persona", markdown="Instructions", expected_version=None
                ),
            )

    assert error.value.status_code == 404
    assert writes(db) == []
