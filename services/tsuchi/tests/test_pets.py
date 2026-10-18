import io
import sqlite3
from contextlib import nullcontext
from datetime import timedelta
from unittest.mock import MagicMock, Mock
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
    PetJob,
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


def artwork(repository, key="alpha", image=True, media_type=None):
    now = utc_now()
    return repository.create_asset(
        Asset(
            id=uuid4(),
            agent_key=key,
            title="Artwork",
            filename="pet.png",
            media_type=media_type or ("image/png" if image else "text/plain"),
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
            raw_asset_id=artwork(repository).id,
            provenance_asset_id=artwork(
                repository, image=False, media_type="application/json"
            ).id,
            source_asset_id=artwork(repository).id,
            prompt=f"Generate {name}",
            request_id=uuid4(),
            model_id="local-edit",
            seed=42,
        )
        for name in PET_STEP_NAMES
    ]
    for completed in range(1, len(steps) + 1):
        pet = repository.update_pet(
            "alpha",
            pet.id,
            state(pet, steps=steps[:completed], completed_steps=completed),
            pet.version,
            pet.owner_id,
        )
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
            completed_steps=len(PET_STEP_NAMES),
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


@pytest.mark.parametrize(
    "reference", ["atlas", "base", "source", "frame", "raw", "provenance"]
)
def test_trashing_pet_dependency_clears_selection_and_blocks_reselection(
    repository, reference
):
    pet = ready_pet(repository)
    repository.select_pet("alpha", pet.id)
    asset_id = {
        "atlas": pet.manifest.atlas_asset_id,
        "base": pet.manifest.base_asset_id,
        "source": pet.steps[1].source_asset_id,
        "frame": pet.steps[1].output_asset_id,
        "raw": pet.steps[1].raw_asset_id,
        "provenance": pet.steps[1].provenance_asset_id,
    }[reference]
    repository.trash_asset("alpha", asset_id, 1)
    assert repository.selected_pet("alpha") is None
    with pytest.raises(
        StorageError,
        match="active JSON" if reference == "provenance" else "active raster",
    ):
        repository.select_pet("alpha", pet.id)
    repository.restore_asset("alpha", asset_id, 2)
    assert repository.selected_pet("alpha") is None


@pytest.mark.parametrize("invalid", ["other_agent", "not_image", "trashed"])
@pytest.mark.parametrize("field", ["output_asset_id", "raw_asset_id"])
def test_step_requires_active_same_agent_raster(repository, invalid, field):
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
        output_asset_id=asset.id
        if field == "output_asset_id"
        else artwork(repository).id,
        raw_asset_id=asset.id if field == "raw_asset_id" else artwork(repository).id,
        provenance_asset_id=artwork(
            repository, image=False, media_type="application/json"
        ).id,
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
                    raw_asset_id=uuid4(),
                    provenance_asset_id=uuid4(),
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


def generation_step(repository, name):
    return PetStep(
        name=name,
        output_asset_id=artwork(repository).id,
        raw_asset_id=artwork(repository).id,
        provenance_asset_id=artwork(
            repository, image=False, media_type="application/json"
        ).id,
        prompt=f"Generate {name}",
        request_id=uuid4(),
        model_id="local-reference",
        seed=42,
    )


def test_completed_frame_survives_cancel_resume_and_rejects_replacement(repository):
    pet = claim(repository, repository.create_pet("alpha", uuid4(), "Fox"))
    for name in ("base", "idle/0"):
        step = generation_step(repository, name)
        pet = repository.update_pet(
            "alpha",
            pet.id,
            state(pet, steps=[*pet.steps, step], completed_steps=len(pet.steps) + 1),
            pet.version,
            pet.owner_id,
        )
    pending_id = uuid4()
    pet = repository.update_pet(
        "alpha",
        pet.id,
        state(pet, pending_step_name="idle/1", pending_request_id=pending_id),
        pet.version,
        pet.owner_id,
    )
    with pytest.raises(StorageError, match="immutable"):
        replacement = pet.steps[-1].model_copy(
            update={"prompt": "Replace the saved frame"}
        )
        repository.update_pet(
            "alpha",
            pet.id,
            state(pet, steps=[pet.steps[0], replacement]),
            pet.version,
            pet.owner_id,
        )
    cancelled = repository.update_pet(
        "alpha",
        pet.id,
        state(
            pet,
            status="cancelled",
            cancel_requested=True,
            owner_id=None,
            lease_expires_at=None,
        ),
        pet.version,
        pet.owner_id,
    )
    resumed = repository.update_pet(
        "alpha",
        pet.id,
        state(cancelled, status="queued", cancel_requested=False),
        cancelled.version,
    )
    reloaded = Repository(repository.data_dir).get_pet("alpha", pet.id)
    assert reloaded == resumed
    assert reloaded.steps == pet.steps
    assert reloaded.completed_steps == 2
    assert reloaded.total_steps == 74
    assert reloaded.pending_step_name == "idle/1"
    assert reloaded.pending_request_id == pending_id
    assert pet.steps[-1].raw_asset_id in reloaded.asset_references()
    assert pet.steps[-1].provenance_asset_id in reloaded.asset_references()


def test_progress_rejects_skipping_frame_persistence(repository):
    pet = claim(repository, repository.create_pet("alpha", uuid4(), "Fox"))
    with pytest.raises(StorageError, match="one completed step"):
        repository.update_pet(
            "alpha",
            pet.id,
            state(
                pet,
                steps=[
                    generation_step(repository, name) for name in ("base", "idle/0")
                ],
                completed_steps=2,
            ),
            pet.version,
            pet.owner_id,
        )
    assert repository.get_pet("alpha", pet.id) == pet


@pytest.mark.parametrize("invalid", ["other_agent", "not_json", "trashed"])
def test_step_provenance_requires_active_same_agent_json(repository, invalid):
    pet = claim(repository, repository.create_pet("alpha", uuid4(), "Fox"))
    provenance = artwork(
        repository,
        key="beta" if invalid == "other_agent" else "alpha",
        image=False,
        media_type="text/plain" if invalid == "not_json" else "application/json",
    )
    if invalid == "trashed":
        repository.trash_asset("alpha", provenance.id, provenance.version)
    step = generation_step(repository, "base").model_copy(
        update={"provenance_asset_id": provenance.id}
    )
    with pytest.raises(StorageError):
        repository.update_pet(
            "alpha",
            pet.id,
            state(pet, steps=[step], completed_steps=1),
            pet.version,
            pet.owner_id,
        )
    assert repository.get_pet("alpha", pet.id) == pet


def test_pending_step_must_identify_next_frame():
    with pytest.raises(ValidationError, match="next unfinished step"):
        PetState(pending_step_name="idle/0", pending_request_id=uuid4())
    assert PET_STEP_NAMES[1:7] == tuple(f"idle/{frame}" for frame in range(6))
    assert PET_STEP_NAMES[-1] == "look-1/7"


@pytest.mark.parametrize("seed", [None, 0, 2**32 - 1])
def test_pending_seed_accepts_random_or_uint32(seed):
    pet = PetState(
        pending_step_name="base", pending_request_id=uuid4(), pending_seed=seed
    )
    assert pet.pending_seed == seed


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_pending_seed_rejects_out_of_range(seed):
    with pytest.raises(ValidationError):
        PetState(
            pending_step_name="base", pending_request_id=uuid4(), pending_seed=seed
        )


def test_pending_seed_requires_request():
    with pytest.raises(ValidationError, match="requires a pending request"):
        PetState(pending_seed=42)


def test_cancellation_and_expired_recovery_cannot_change_pending_seed(
    repository, monkeypatch
):
    pet = claim(repository, repository.create_pet("alpha", uuid4(), "Fox"))
    pet = repository.update_pet(
        "alpha",
        pet.id,
        state(
            pet, pending_step_name="base", pending_request_id=uuid4(), pending_seed=42
        ),
        pet.version,
        pet.owner_id,
    )
    with pytest.raises(StorageError, match="Cancellation cannot modify"):
        repository.update_pet(
            "alpha",
            pet.id,
            state(pet, status="cancelling", cancel_requested=True, pending_seed=43),
            pet.version,
        )
    monkeypatch.setattr(
        "tsuchi.repository.utc_now", lambda: pet.lease_expires_at + timedelta(seconds=1)
    )
    with pytest.raises(StorageError, match="Recovery cannot modify"):
        repository.update_pet(
            "alpha",
            pet.id,
            state(
                pet,
                status="interrupted",
                owner_id=None,
                lease_expires_at=None,
                pending_seed=43,
            ),
            pet.version,
        )
    interrupted = repository.update_pet(
        "alpha",
        pet.id,
        state(pet, status="interrupted", owner_id=None, lease_expires_at=None),
        pet.version,
    )
    assert interrupted.pending_seed == 42
    assert interrupted.pending_request_id == pet.pending_request_id


def test_revision_keeps_other_frames_history_and_selected_original(repository):
    original = ready_pet(repository)
    repository.select_pet("alpha", original.id)
    revision_id = uuid4()
    repaired = repository.revise_pet(
        "alpha",
        original.id,
        revision_id,
        original.version,
        ["idle/2", "running-left/3"],
        "Clipped artwork",
    )
    assert repaired.status == "queued"
    assert repaired.parent_pet_id == original.id
    assert repaired.manifest is None
    assert repaired.pending_request_id is None
    assert repaired.steps == [
        step for step in original.steps if step.name not in {"idle/2", "running-left/3"}
    ]
    assert [item.step.name for item in repaired.rejections] == [
        "idle/2",
        "running-left/3",
    ]
    assert all(item.reason == "Clipped artwork" for item in repaired.rejections)
    assert (
        original.asset_references() - {original.manifest.atlas_asset_id}
        <= repaired.asset_references()
    )
    assert repository.get_pet("alpha", original.id) == original
    assert repository.selected_pet("alpha") == original
    assert (
        repository.revise_pet(
            "alpha",
            original.id,
            revision_id,
            original.version,
            ["running-left/3", "idle/2"],
            "Clipped artwork",
        )
        == repaired
    )
    with pytest.raises(StorageError, match="already in use"):
        repository.revise_pet(
            "alpha",
            original.id,
            revision_id,
            original.version,
            ["idle/2"],
            "Clipped artwork",
        )
    active = claim(repository, repaired)
    next_request = uuid4()
    active = repository.update_pet(
        "alpha",
        active.id,
        state(
            active,
            pending_step_name="idle/2",
            pending_request_id=next_request,
            pending_seed=43,
        ),
        active.version,
        active.owner_id,
    )
    replacement = generation_step(repository, "idle/2")
    completed = sorted(
        [*active.steps, replacement], key=lambda step: PET_STEP_NAMES.index(step.name)
    )
    active = repository.update_pet(
        "alpha",
        active.id,
        state(
            active,
            steps=completed,
            completed_steps=73,
            pending_step_name=None,
            pending_request_id=None,
            pending_seed=None,
        ),
        active.version,
        active.owner_id,
    )
    assert active.steps[3] == replacement
    assert all(step in active.steps for step in repaired.steps)
    assert active.rejections == repaired.rejections
    with repository.connect() as db:
        exported = repository.export_manifest(db, "alpha")
    assert active in exported.pet_jobs
    assert active.asset_references() <= {asset.id for asset in exported.assets}
    stopped = repository.update_pet(
        "alpha",
        active.id,
        state(active, status="failed", owner_id=None, lease_expires_at=None),
        active.version,
        active.owner_id,
    )
    second = repository.revise_pet(
        "alpha", stopped.id, uuid4(), stopped.version, ["idle/2"], "Wrong pose"
    )
    assert second.rejections[:2] == repaired.rejections
    assert second.rejections[-1].step == replacement
    assert repository.selected_pet("alpha") == original


@pytest.mark.parametrize(
    "invalid",
    ["active", "namespace", "version", "base", "missing", "duplicate", "blank"],
)
def test_revision_rejects_invalid_source_or_request(repository, invalid):
    original = ready_pet(repository)
    pet_id, key, version = original.id, "alpha", original.version
    names, reason = ["idle/2"], "Wrong pose"
    if invalid == "active":
        active = repository.create_pet("alpha", uuid4(), "Fox")
        pet_id, version = active.id, active.version
    elif invalid == "namespace":
        key = "beta"
    elif invalid == "version":
        version -= 1
    elif invalid == "base":
        names = ["base"]
    elif invalid == "missing":
        names = ["idle/99"]
    elif invalid == "duplicate":
        names = ["idle/2", "idle/2"]
    else:
        reason = "  "
    with pytest.raises(StorageError):
        repository.revise_pet(key, pet_id, uuid4(), version, names, reason)
    assert repository.get_pet("alpha", original.id) == original


def test_worker_cannot_rewrite_rejection_history(repository):
    original = ready_pet(repository)
    revision = repository.revise_pet(
        "alpha", original.id, uuid4(), original.version, ["idle/2"], "Wrong pose"
    )
    active = claim(repository, revision)
    with pytest.raises(StorageError, match="history is immutable"):
        repository.update_pet(
            "alpha",
            active.id,
            state(active, rejections=[]),
            active.version,
            active.owner_id,
        )
    with pytest.raises(StorageError, match="history is immutable"):
        repository.update_pet(
            "alpha",
            active.id,
            state(active, parent_pet_id=None),
            active.version,
            active.owner_id,
        )


@pytest.mark.parametrize("field", ["raw_asset_id", "provenance_asset_id"])
def test_rejected_artwork_remains_validated_dependency(repository, field):
    original = ready_pet(repository)
    revision = repository.revise_pet(
        "alpha", original.id, uuid4(), original.version, ["idle/2"], "Wrong pose"
    )
    active = claim(repository, revision)
    stopped = repository.update_pet(
        "alpha",
        active.id,
        state(active, status="failed", owner_id=None, lease_expires_at=None),
        active.version,
        active.owner_id,
    )
    asset_id = getattr(stopped.rejections[0].step, field)
    repository.trash_asset("alpha", asset_id, 1)
    with pytest.raises(StorageError, match="active"):
        repository.revise_pet(
            "alpha", stopped.id, uuid4(), stopped.version, ["idle/3"], "Wrong pose"
        )


def test_canonical_frames_require_base_and_idle_endpoints(repository):
    base = generation_step(repository, "base")
    frame = generation_step(repository, "idle/0").model_copy(
        update={"method": "canonical", "source_asset_id": base.output_asset_id}
    )
    pet = PetState(steps=[base, frame], completed_steps=2)
    assert pet.steps[1].method == "canonical"
    for invalid in [
        frame.model_copy(update={"name": "idle/1"}),
        frame.model_copy(update={"source_asset_id": uuid4()}),
    ]:
        with pytest.raises(ValidationError, match="Canonical frames"):
            PetState(steps=[base, invalid], completed_steps=2)
    with pytest.raises(ValidationError, match="canonical creation order"):
        PetState(steps=[frame], completed_steps=1)


def test_pending_cohort_supports_out_of_order_checkpoints_and_survives_resume(
    repository,
):
    pet = claim(repository, repository.create_pet("alpha", uuid4(), "Fox"))
    for name in ("base", "idle/0"):
        pet = repository.update_pet(
            "alpha",
            pet.id,
            state(
                pet,
                steps=[*pet.steps, generation_step(repository, name)],
                completed_steps=len(pet.steps) + 1,
            ),
            pet.version,
            pet.owner_id,
        )
    first, second, third = uuid4(), uuid4(), uuid4()
    pet = repository.update_pet(
        "alpha",
        pet.id,
        state(
            pet,
            pending_step_name="idle/1",
            pending_request_id=first,
            pending_seed=10,
            additional_pending_frames=[
                {"name": "idle/2", "request_id": second, "seed": 11},
                {"name": "idle/3", "request_id": third, "seed": 12},
            ],
        ),
        pet.version,
        pet.owner_id,
    )
    completed = generation_step(repository, "idle/3").model_copy(
        update={"request_id": third}
    )
    pet = repository.update_pet(
        "alpha",
        pet.id,
        state(
            pet,
            steps=[*pet.steps, completed],
            completed_steps=3,
            additional_pending_frames=pet.additional_pending_frames[:1],
        ),
        pet.version,
        pet.owner_id,
    )
    assert [frame.name for frame in pet.pending_images] == ["idle/1", "idle/2"]
    assert pet.steps[-1] == completed
    with pytest.raises(StorageError, match="Cancellation cannot modify"):
        repository.update_pet(
            "alpha",
            pet.id,
            state(
                pet,
                status="cancelling",
                cancel_requested=True,
                additional_pending_frames=[],
            ),
            pet.version,
        )
    stopped = repository.update_pet(
        "alpha",
        pet.id,
        state(
            pet,
            status="cancelled",
            cancel_requested=True,
            owner_id=None,
            lease_expires_at=None,
        ),
        pet.version,
        pet.owner_id,
    )
    resumed = repository.update_pet(
        "alpha",
        pet.id,
        state(stopped, status="queued", cancel_requested=False),
        stopped.version,
    )
    reloaded = Repository(repository.data_dir).get_pet("alpha", pet.id)
    assert reloaded == resumed
    assert reloaded.pending_images == pet.pending_images
    assert reloaded.steps == pet.steps


@pytest.mark.parametrize(
    "invalid",
    [
        "duplicate_name",
        "duplicate_request",
        "completed",
        "unordered",
        "canonical",
        "oversized",
    ],
)
def test_pending_cohort_rejects_invalid_members(repository, invalid):
    base = generation_step(repository, "base")
    neutral = generation_step(repository, "idle/0")
    primary = uuid4()
    pending = [
        {"name": "idle/2", "request_id": uuid4(), "seed": 42},
        {"name": "idle/3", "request_id": uuid4(), "seed": 42},
    ]
    if invalid == "duplicate_name":
        pending[0]["name"] = "idle/1"
    elif invalid == "duplicate_request":
        pending[0]["request_id"] = primary
    elif invalid == "completed":
        pending[0]["name"] = "idle/0"
    elif invalid == "unordered":
        pending.reverse()
    elif invalid == "canonical":
        pending[1]["name"] = "idle/5"
    else:
        pending.extend(
            [
                {"name": "idle/4", "request_id": uuid4()},
                {"name": "walk_right/0", "request_id": uuid4()},
            ]
        )
    with pytest.raises(ValidationError):
        PetState(
            steps=[base, neutral],
            completed_steps=2,
            pending_step_name="idle/1",
            pending_request_id=primary,
            additional_pending_frames=pending,
        )


def test_additional_pending_frames_require_a_primary_request():
    with pytest.raises(ValidationError, match="primary"):
        PetState(additional_pending_frames=[{"name": "idle/2", "request_id": uuid4()}])


def imported_manifest(atlas_id, sprite_version=2):
    return PetManifest(
        atlas_asset_id=atlas_id,
        sprite_version=sprite_version,
        rows=11 if sprite_version == 2 else 9,
        animations=[
            PetAnimation(name=name, row=row, durations=[100] * 4)
            for row, name in enumerate(PET_ANIMATIONS)
        ],
    )


@pytest.fixture
def imported_pet():
    now = utc_now()
    return PetJob(
        id=uuid4(),
        agent_key="alpha",
        prompt="Copper fox",
        origin="imported",
        status="ready",
        stage="Imported",
        manifest=imported_manifest(uuid4()),
        version=1,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def imported_repository(tmp_path, monkeypatch, imported_pet):
    repository = Repository(tmp_path)
    db = MagicMock(spec=sqlite3.Connection)
    db.execute.return_value.fetchone.return_value = None
    monkeypatch.setattr(repository, "connect", Mock(return_value=nullcontext(db)))
    monkeypatch.setattr(repository, "_agent", Mock())
    monkeypatch.setattr(repository, "_pet", Mock(return_value=imported_pet))
    monkeypatch.setattr(repository, "_save_pet", Mock())
    monkeypatch.setattr(
        repository,
        "_asset",
        Mock(
            return_value=Asset(
                id=imported_pet.manifest.atlas_asset_id,
                agent_key="alpha",
                title="Atlas",
                filename="pet.png",
                media_type="image/png",
                size_bytes=10,
                sha256="digest",
                is_image=True,
                version=1,
                created_at=imported_pet.created_at,
                updated_at=imported_pet.updated_at,
            )
        ),
    )
    return repository, db


@pytest.mark.parametrize("sprite_version", [1, 2])
def test_imported_pet_is_saved_ready(imported_repository, imported_pet, sprite_version):
    repository, db = imported_repository
    manifest = imported_manifest(imported_pet.manifest.atlas_asset_id, sprite_version)
    pet = repository.import_pet("alpha", imported_pet.id, "Copper fox", manifest)
    assert (pet.origin, pet.status, pet.steps, pet.manifest) == (
        "imported",
        "ready",
        [],
        manifest,
    )
    assert (pet.id, pet.agent_key, pet.prompt, pet.version) == (
        imported_pet.id,
        "alpha",
        "Copper fox",
        1,
    )
    assert pet.asset_references() == {manifest.atlas_asset_id}
    repository._asset.assert_called_once_with(db, "alpha", manifest.atlas_asset_id)
    repository._save_pet.assert_called_once_with(db, pet)


def test_imported_pet_is_idempotent(imported_repository, imported_pet):
    repository, db = imported_repository
    db.execute.return_value.fetchone.return_value = (imported_pet.model_dump_json(),)
    assert (
        repository.import_pet(
            "alpha", imported_pet.id, imported_pet.prompt, imported_pet.manifest
        )
        == imported_pet
    )
    repository._save_pet.assert_not_called()


@pytest.mark.parametrize("conflict", ["agent", "prompt", "manifest"])
def test_imported_pet_rejects_id_conflicts(imported_repository, imported_pet, conflict):
    repository, db = imported_repository
    db.execute.return_value.fetchone.return_value = (imported_pet.model_dump_json(),)
    with pytest.raises(StorageError, match="already in use"):
        repository.import_pet(
            "beta" if conflict == "agent" else "alpha",
            imported_pet.id,
            "Another fox" if conflict == "prompt" else imported_pet.prompt,
            imported_manifest(uuid4())
            if conflict == "manifest"
            else imported_pet.manifest,
        )
    repository._save_pet.assert_not_called()


@pytest.mark.parametrize("invalid", ["other_agent", "not_image", "deleted"])
def test_imported_pet_requires_same_agent_raster_atlas(
    imported_repository, imported_pet, invalid
):
    repository, db = imported_repository
    if invalid == "other_agent":
        repository._asset.side_effect = StorageError(
            "Asset not found in this namespace", 404
        )
        message = "not found in this namespace"
    else:
        changes = (
            {"is_image": False, "media_type": "text/plain"}
            if invalid == "not_image"
            else {"deleted_at": utc_now()}
        )
        repository._asset.return_value = repository._asset.return_value.model_copy(
            update=changes
        )
        message = "active raster images"
    with pytest.raises(StorageError, match=message):
        repository.import_pet(
            "alpha", imported_pet.id, imported_pet.prompt, imported_pet.manifest
        )
    repository._asset.assert_called_once_with(
        db, "alpha", imported_pet.manifest.atlas_asset_id
    )
    repository._save_pet.assert_not_called()


def test_imported_pet_can_be_selected(imported_repository, imported_pet):
    repository, db = imported_repository
    assert repository.select_pet("alpha", imported_pet.id) == imported_pet
    repository._asset.assert_called_once_with(
        db, "alpha", imported_pet.manifest.atlas_asset_id
    )
    db.execute.assert_called_once_with(
        "UPDATE agents SET selected_pet_id = ? WHERE key = ?",
        (str(imported_pet.id), "alpha"),
    )


def test_imported_pet_cannot_be_resumed(imported_repository, imported_pet):
    repository, _ = imported_repository
    with pytest.raises(StorageError, match="transition"):
        repository.update_pet(
            "alpha",
            imported_pet.id,
            state(
                imported_pet,
                origin="generated",
                status="queued",
                manifest=None,
                stage="q",
            ),
            1,
        )
    repository._save_pet.assert_not_called()


def test_imported_pet_cannot_be_revised(imported_repository, imported_pet):
    repository, _ = imported_repository
    with pytest.raises(StorageError, match="completed frames"):
        repository.revise_pet(
            "alpha", imported_pet.id, uuid4(), 1, ["idle/1"], "Fix it"
        )
    repository._save_pet.assert_not_called()


def test_imported_pet_cannot_carry_generation_history():
    with pytest.raises(ValidationError, match="Imported pets"):
        PetState(origin="imported", status="ready", manifest=None)
    with pytest.raises(ValidationError, match="Imported pets"):
        PetState(
            origin="imported",
            status="ready",
            manifest=imported_manifest(uuid4()).model_copy(
                update={"base_asset_id": uuid4()}
            ),
        )


def test_manifest_rows_match_sprite_version():
    with pytest.raises(ValidationError, match="9 rows"):
        PetManifest.model_validate(
            {**imported_manifest(uuid4()).model_dump(), "sprite_version": 1}
        )
