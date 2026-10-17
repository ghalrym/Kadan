import io
import sqlite3
from datetime import timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError
from tsuchi.config import StorageError
from tsuchi.models import (
    PET_ANIMATIONS,
    PET_STEP_NAMES,
    AgentInput,
    Asset,
    Metadata,
    PetAnimation,
    PetManifest,
    PetState,
    PetStep,
    utc_now,
)
from tsuchi.repository import Repository
from tsuchi.services import FileStorage


@pytest.fixture
def repository(tmp_path):
    repository = Repository(tmp_path)
    repository.initialize()
    repository.create_agent(AgentInput(key="alpha", name="Alpha"))
    repository.create_agent(AgentInput(key="beta", name="Beta"))
    return repository


def state(pet, **changes):
    return PetState.model_validate(
        {
            **pet.model_dump(include=set(PetState.model_fields)),
            **changes,
        }
    )


def artwork(repository, key="alpha", image=True):
    now = utc_now()
    return repository.create_asset(
        Asset(
            id=uuid4(),
            agent_key=key,
            title="Artwork",
            filename="pet.png",
            media_type="image/png" if image else "text/plain",
            size_bytes=10,
            sha256="digest",
            is_image=image,
            version=1,
            created_at=now,
            updated_at=now,
        )
    )


def claim(repository, pet):
    owner = uuid4()
    return repository.update_pet(
        "alpha",
        pet.id,
        state(
            pet,
            status="running",
            stage="base",
            owner_id=owner,
            lease_expires_at=utc_now() + timedelta(minutes=2),
        ),
        pet.version,
        owner,
    )


def ready_pet(repository):
    pet = claim(repository, repository.create_pet("alpha", uuid4(), "Watercolor fox"))
    steps = [
        PetStep(
            name=name,
            output_asset_id=artwork(repository).id,
            source_asset_id=artwork(repository).id,
            prompt=f"Generate {name}",
            request_id=uuid4(),
            model_id="local-edit",
            seed=42,
        )
        for name in PET_STEP_NAMES
    ]
    manifest = PetManifest(
        atlas_asset_id=artwork(repository).id,
        base_asset_id=steps[0].output_asset_id,
        animations=[
            PetAnimation(name=name, row=row, durations=[100] * 8)
            for row, name in enumerate(PET_ANIMATIONS)
        ],
    )
    return repository.update_pet(
        "alpha",
        pet.id,
        state(
            pet,
            status="ready",
            stage="ready",
            steps=steps,
            completed_steps=12,
            manifest=manifest,
            owner_id=None,
            lease_expires_at=None,
        ),
        pet.version,
        pet.owner_id,
    )


def test_existing_database_migrates_without_losing_agents(tmp_path):
    with sqlite3.connect(tmp_path / "tsuchi.sqlite3") as db:
        db.execute(
            "CREATE TABLE agents (key TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        db.execute(
            "INSERT INTO agents VALUES (?, ?, ?)",
            ("alpha", "Alpha", utc_now().isoformat()),
        )
    repository = Repository(tmp_path)
    repository.initialize()
    repository.initialize()
    assert repository.get_agent("alpha").selected_pet_id is None
    assert repository.list_pets("alpha") == []


def test_creation_idempotent_and_scoped(repository):
    pet = repository.create_pet("alpha", uuid4(), "A fox")
    assert repository.create_pet("alpha", pet.id, "A fox") == pet
    for key, prompt in [("beta", "A fox"), ("alpha", "A turtle")]:
        with pytest.raises(StorageError, match="already in use"):
            repository.create_pet(key, pet.id, prompt)
    with pytest.raises(StorageError, match="not found"):
        repository.get_pet("beta", pet.id)
    assert repository.selected_pet("alpha") is None


def test_ready_pet_persists_and_exports_without_auto_activation(repository):
    pet = ready_pet(repository)
    assert repository.selected_pet("alpha") is None
    reloaded = Repository(repository.data_dir)
    assert reloaded.get_pet("alpha", pet.id) == pet
    with reloaded.connect() as db:
        exported = reloaded.export_manifest(db, "alpha")
    assert exported.pet_jobs == [pet]
    assert pet.asset_references() <= {asset.id for asset in exported.assets}
    assert reloaded.select_pet("alpha", pet.id) == pet
    assert reloaded.get_agent("alpha").selected_pet_id == pet.id


def test_selection_requires_ready_same_agent_and_generation_preserves_selection(
    repository,
):
    selected = ready_pet(repository)
    repository.select_pet("alpha", selected.id)
    pending = repository.create_pet("alpha", uuid4(), "Another pet")
    with pytest.raises(StorageError, match="Only ready"):
        repository.select_pet("alpha", pending.id)
    with pytest.raises(StorageError, match="not found"):
        repository.select_pet("beta", selected.id)
    pending = claim(repository, pending)
    repository.update_pet(
        "alpha",
        pending.id,
        state(
            pending,
            status="failed",
            error="Generation failed",
            owner_id=None,
            lease_expires_at=None,
        ),
        pending.version,
        pending.owner_id,
    )
    assert repository.selected_pet("alpha") == selected


@pytest.mark.parametrize("reference", ["atlas", "base", "source", "strip"])
def test_trashing_pet_dependency_clears_selection_and_blocks_reselection(
    repository, reference
):
    pet = ready_pet(repository)
    repository.select_pet("alpha", pet.id)
    asset_id = {
        "atlas": pet.manifest.atlas_asset_id,
        "base": pet.manifest.base_asset_id,
        "source": pet.steps[1].source_asset_id,
        "strip": pet.steps[1].output_asset_id,
    }[reference]
    repository.trash_asset("alpha", asset_id, 1)
    assert repository.selected_pet("alpha") is None
    with pytest.raises(StorageError, match="active raster"):
        repository.select_pet("alpha", pet.id)
    repository.restore_asset("alpha", asset_id, 2)
    assert repository.selected_pet("alpha") is None


@pytest.mark.parametrize("invalid", ["other_agent", "not_image", "trashed"])
def test_step_requires_active_same_agent_raster(repository, invalid):
    pet = claim(repository, repository.create_pet("alpha", uuid4(), "Fox"))
    asset = artwork(
        repository,
        key="beta" if invalid == "other_agent" else "alpha",
        image=invalid != "not_image",
    )
    if invalid == "trashed":
        repository.trash_asset("alpha", asset.id, asset.version)
    step = PetStep(
        name="base",
        output_asset_id=asset.id,
        prompt="Fox",
        request_id=uuid4(),
        model_id="local",
    )
    with pytest.raises(StorageError):
        repository.update_pet(
            "alpha",
            pet.id,
            state(pet, steps=[step], completed_steps=1),
            pet.version,
            pet.owner_id,
        )
    assert repository.get_pet("alpha", pet.id).completed_steps == 0


def test_stale_worker_cannot_claim_or_overwrite_job(repository):
    queued = repository.create_pet("alpha", uuid4(), "Fox")
    pet = claim(repository, queued)
    with pytest.raises(StorageError, match="changed"):
        claim(repository, queued)
    with pytest.raises(StorageError, match="not owned"):
        repository.update_pet(
            "alpha", pet.id, state(pet, stage="overwrite"), pet.version, uuid4()
        )
    assert repository.get_pet("alpha", pet.id) == pet


def test_cancel_is_sticky_and_can_heartbeat_until_worker_stops(repository):
    pet = claim(repository, repository.create_pet("alpha", uuid4(), "Fox"))
    pet = repository.update_pet(
        "alpha",
        pet.id,
        state(
            pet,
            status="cancelling",
            cancel_requested=True,
        ),
        pet.version,
    )
    pet = repository.update_pet(
        "alpha",
        pet.id,
        state(
            pet,
            lease_expires_at=utc_now() + timedelta(minutes=2),
        ),
        pet.version,
        pet.owner_id,
    )
    with pytest.raises(StorageError, match="cannot be cleared"):
        repository.update_pet(
            "alpha",
            pet.id,
            state(
                pet,
                status="failed",
                cancel_requested=False,
                owner_id=None,
                lease_expires_at=None,
            ),
            pet.version,
            pet.owner_id,
        )
    stopped = repository.update_pet(
        "alpha",
        pet.id,
        state(
            pet,
            status="cancelled",
            owner_id=None,
            lease_expires_at=None,
        ),
        pet.version,
        pet.owner_id,
    )
    assert stopped.cancel_requested
    assert repository.active_pet_jobs() == []


def test_expired_job_is_interrupted_and_requires_explicit_resume(
    repository, monkeypatch
):
    pet = claim(repository, repository.create_pet("alpha", uuid4(), "Fox"))
    pending = uuid4()
    pet = repository.update_pet(
        "alpha",
        pet.id,
        state(
            pet,
            pending_step_name="base",
            pending_request_id=pending,
        ),
        pet.version,
        pet.owner_id,
    )
    monkeypatch.setattr(
        "tsuchi.repository.utc_now", lambda: pet.lease_expires_at + timedelta(seconds=1)
    )
    with pytest.raises(StorageError, match="not owned"):
        repository.update_pet(
            "alpha",
            pet.id,
            state(pet, stage="still working"),
            pet.version,
            pet.owner_id,
        )
    interrupted = repository.update_pet(
        "alpha",
        pet.id,
        state(
            pet,
            status="interrupted",
            owner_id=None,
            lease_expires_at=None,
        ),
        pet.version,
    )
    assert interrupted.pending_request_id == pending
    assert repository.active_pet_jobs() == []
    queued = repository.update_pet(
        "alpha", pet.id, state(interrupted, status="queued"), interrupted.version
    )
    assert queued.pending_request_id == pending
    assert repository.active_pet_jobs() == [queued]


def test_cancellation_cannot_change_progress(repository):
    pet = claim(repository, repository.create_pet("alpha", uuid4(), "Fox"))
    with pytest.raises(StorageError, match="cannot modify"):
        repository.update_pet(
            "alpha",
            pet.id,
            state(
                pet,
                status="cancelling",
                cancel_requested=True,
                pending_step_name="base",
                pending_request_id=uuid4(),
            ),
            pet.version,
        )


def test_pet_manifest_rejects_missing_or_wrong_animation_rows():
    with pytest.raises(ValidationError, match="canonical"):
        PetManifest(
            atlas_asset_id=uuid4(),
            base_asset_id=uuid4(),
            animations=[
                PetAnimation(name="idle", row=0, durations=[100]) for _ in range(9)
            ],
        )
    with pytest.raises(ValidationError, match="every step"):
        PetState(status="ready")
    with pytest.raises(ValidationError, match="canonical creation order"):
        PetState(
            steps=[
                PetStep(
                    name="waving",
                    output_asset_id=uuid4(),
                    prompt="wave",
                    request_id=uuid4(),
                    model_id="local",
                )
            ],
            completed_steps=1,
        )


def test_idempotent_asset_upload_preserves_first_record_and_bytes(repository):
    files = FileStorage(repository, max_upload_bytes=100)
    files.initialize()
    asset_id = uuid4()
    args = ("file.txt", Metadata(title="File"), asset_id)
    asset = files.upload("alpha", io.BytesIO(b"abc"), *args)
    assert files.upload("alpha", io.BytesIO(b"abc"), *args) == asset
    with pytest.raises(StorageError, match="different artwork"):
        files.upload("alpha", io.BytesIO(b"xyz"), *args)
    with pytest.raises(StorageError, match="already in use"):
        files.upload("beta", io.BytesIO(b"abc"), *args)
    assert files.blob_path(asset_id).read_bytes() == b"abc"
    assert len(repository.list_assets("alpha")) == 1


def test_asset_retry_recovers_blob_published_before_database_write(
    repository, monkeypatch
):
    files = FileStorage(repository, max_upload_bytes=100)
    files.initialize()
    asset_id = uuid4()
    save = repository.create_asset

    def failed_write(asset):
        raise RuntimeError("Database interrupted")

    monkeypatch.setattr(repository, "create_asset", failed_write)
    with pytest.raises(RuntimeError):
        files.upload(
            "alpha", io.BytesIO(b"abc"), "file.txt", Metadata(title="File"), asset_id
        )
    monkeypatch.setattr(repository, "create_asset", save)
    asset = files.upload(
        "alpha", io.BytesIO(b"abc"), "file.txt", Metadata(title="File"), asset_id
    )
    assert asset.id == asset_id
    assert files.blob_path(asset_id).read_bytes() == b"abc"


def test_expired_cancellation_can_finish_after_owner_restart(repository, monkeypatch):
    pet = claim(repository, repository.create_pet("alpha", uuid4(), "Fox"))
    pet = repository.update_pet(
        "alpha",
        pet.id,
        state(
            pet,
            status="cancelling",
            cancel_requested=True,
            pending_step_name=None,
            pending_request_id=None,
        ),
        pet.version,
    )
    monkeypatch.setattr(
        "tsuchi.repository.utc_now", lambda: pet.lease_expires_at + timedelta(seconds=1)
    )
    cancelled = repository.update_pet(
        "alpha",
        pet.id,
        state(
            pet,
            status="cancelled",
            owner_id=None,
            lease_expires_at=None,
        ),
        pet.version,
    )
    assert cancelled.status == "cancelled"
    assert repository.active_pet_jobs() == []
