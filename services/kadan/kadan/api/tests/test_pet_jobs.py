import asyncio
import base64
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4, uuid5

import pytest

from kadan.api.config import ServiceError
from kadan.api.pet_artwork import ROWS, SOURCE_CELL
from kadan.api.pet_client import PetClient
from kadan.api.pydantic_models.pets import (
    GeneratedImage,
    PetJob,
    PetPendingFrame,
    PetStep,
)
from kadan.runner.pet_jobs import IMAGE_MODEL, PetJobs

KEY = "andrew"
PET_ID = UUID("10000000-0000-4000-8000-000000000003")
PNG = b"\x89PNG\r\n\x1a\nfixture"


class Storage:
    def __init__(self):
        self.jobs = {}
        self.calls = []
        self.client = SimpleNamespace()
        self.settings = SimpleNamespace(
            hikari_url="http://localhost:8088", tsuchi_url="http://localhost:8620"
        )

    async def json_request(self, service, method, path, **kwargs):
        self.calls.append((service, method, path, deepcopy(kwargs)))
        if service == "Hikari":
            return {"status": "not_active"}
        if method == "POST":
            body = kwargs["json"]
            job_id = UUID(body["id"])
            if job_id not in self.jobs:
                self.jobs[job_id] = PetJob(
                    id=job_id,
                    agent_key=KEY,
                    prompt=body["prompt"],
                    reference_mode=body.get("reference_mode", "single_image"),
                    total_steps=75
                    if body.get("reference_mode") == "character_sheet"
                    else 74,
                    version=1,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            return self.jobs[job_id].model_dump(mode="json")
        job_id = UUID(path.rsplit("/", 1)[1])
        job = self.jobs[job_id]
        if method == "PUT":
            body = kwargs["json"]
            if body["expected_version"] != job.version:
                raise ServiceError("Conflict", 409)
            values = {
                k: v
                for k, v in body.items()
                if k not in {"worker_id", "expected_version"}
            }
            job = PetJob.model_validate(
                {**job.model_dump(), **values, "version": job.version + 1}
            )
            self.jobs[job_id] = job
        return job.model_dump(mode="json")


async def setup_running(reference_mode="single_image"):
    services = Storage()
    pets = PetJobs(services)
    job = await pets.create(KEY, PET_ID, "An ink-painted owl", reference_mode)
    job = await pets.save(
        job,
        worker_id=pets.owner,
        status="running",
        owner_id=pets.owner,
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
    )
    return pets, services, job


def test_create_persists_idempotent_job_without_inference_or_selection():
    async def run():
        services = Storage()
        pets = PetClient(services)
        one = await pets.create(KEY, PET_ID, "A fox")
        two = await pets.create(KEY, PET_ID, "A fox")
        assert one.id == two.id == PET_ID
        assert one.status == "queued"
        assert all(
            call[0] == "Tsuchi" and call[2].endswith("/pets") for call in services.calls
        )
        assert not hasattr(pets, "wakeup")

    asyncio.run(run())


def test_cancel_waits_for_worker_cleanup_before_terminal_status():
    async def run():
        pets, services, job = await setup_running()
        cancelled = await pets.cancel(KEY, job.id)
        assert cancelled.status == "cancelling"
        assert cancelled.owner_id == pets.owner
        assert cancelled.cancel_requested
        assert not any(call[0] == "Hikari" for call in services.calls)

    asyncio.run(run())


def test_cancel_queued_job_needs_no_generation_cleanup():
    async def run():
        services = Storage()
        pets = PetJobs(services)
        await pets.create(KEY, PET_ID, "A fox")
        job = await pets.cancel(KEY, PET_ID)
        assert job.status == "cancelled"
        assert job.owner_id is None

    asyncio.run(run())


def test_resume_fences_old_request_before_requeueing():
    async def run():
        pets, services, job = await setup_running()
        request_id = uuid4()
        job = await pets.save(
            job,
            worker_id=pets.owner,
            status="interrupted",
            owner_id=None,
            lease_expires_at=None,
            pending_step_name="base",
            pending_request_id=request_id,
        )
        resumed = await pets.resume(KEY, job.id)
        assert resumed.status == "queued"
        assert resumed.pending_request_id == request_id
        assert services.calls[-2][0:3] == (
            "Hikari",
            "POST",
            f"/v1/images/requests/{request_id}/cancel",
        )

    asyncio.run(run())


def test_resume_does_not_claim_recovery_when_cancel_confirmation_fails():
    async def run():
        pets, services, job = await setup_running()
        await pets.save(
            job,
            worker_id=pets.owner,
            status="interrupted",
            owner_id=None,
            lease_expires_at=None,
            pending_step_name="base",
            pending_request_id=uuid4(),
        )
        pets.stop_request = AsyncMock(side_effect=ServiceError("Hikari unavailable"))
        with pytest.raises(ServiceError):
            await pets.resume(KEY, job.id)
        assert services.jobs[job.id].status == "interrupted"

    asyncio.run(run())


def test_worker_cannot_write_after_losing_ownership():
    async def run():
        pets, services, job = await setup_running()
        services.jobs[job.id].owner_id = uuid4()
        with pytest.raises(ServiceError, match="ownership"):
            await pets.owned_update(job, stage="Ready")
        assert services.jobs[job.id].stage == "queued"

    asyncio.run(run())


def test_cleanup_failure_retains_cancelling_instead_of_claiming_cancelled():
    async def run():
        pets, services, job = await setup_running()
        await pets.save(
            job,
            worker_id=pets.owner,
            pending_step_name="base",
            pending_request_id=uuid4(),
        )
        await pets.cancel(KEY, job.id)
        pets.generate = AsyncMock(side_effect=asyncio.CancelledError)
        pets.stop_request = AsyncMock(side_effect=ServiceError("cleanup unknown"))
        await pets.process(job)
        assert services.jobs[job.id].status == "cancelling"

    asyncio.run(run())


@pytest.mark.parametrize("reference_mode", ["single_image", "character_sheet"])
def test_pipeline_uses_selected_reference_and_never_selects(
    monkeypatch, reference_mode
):
    async def inline_thread(function, *args):
        return function(*args)

    monkeypatch.setattr("kadan.runner.pet_jobs.asyncio.to_thread", inline_thread)

    async def run():
        pets, services, job = await setup_running(reference_mode)
        assets = {}
        sources = []
        seeds = []

        async def upload(job, asset_id, label, data, media_type):
            assets[asset_id] = data
            return asset_id

        async def content(key, asset_id):
            return assets[asset_id]

        async def image(
            request_id,
            prompt,
            source,
            model_id,
            seed=None,
            *,
            batch_collection_target=None,
        ):
            sources.append(source)
            seeds.append(seed)
            return GeneratedImage(
                model_id=IMAGE_MODEL,
                seeds=[1],
                images_base64=[base64.b64encode(PNG).decode()],
                mime_type="image/png",
            )

        pets.upload = upload
        pets.content = content
        pets.image = image
        monkeypatch.setattr(
            "kadan.runner.pet_jobs.prepare_base", lambda raw: b"canonical-base"
        )
        monkeypatch.setattr(
            "kadan.runner.pet_jobs.prepare_character_sheet",
            lambda raw: b"character-sheet",
        )
        monkeypatch.setattr(
            "kadan.runner.pet_jobs.prepare_frame", lambda raw, row, frame: PNG
        )

        def assemble(frames):
            assert len(frames) == 73
            assert "base" not in frames and "character-sheet" not in frames
            return b"webp-atlas"

        monkeypatch.setattr("kadan.runner.pet_jobs.assemble_atlas", assemble)
        monkeypatch.setattr(
            "kadan.runner.pet_jobs.validate_row", lambda row, frames: None
        )
        await asyncio.wait_for(pets.generate(job), timeout=5)
        saved = services.jobs[job.id]
        assert saved.status == "ready"
        sheet_mode = reference_mode == "character_sheet"
        assert saved.completed_steps == saved.total_steps == (75 if sheet_mode else 74)
        assert len(saved.steps) == saved.total_steps
        assert len(sources) == (73 if sheet_mode else 72) and sources[0] is None
        reference_bytes = b"character-sheet" if sheet_mode else b"canonical-base"
        assert all(source == reference_bytes for source in sources[1:])
        assert seeds == ([None] * (2 if sheet_mode else 1)) + [1] * 71
        base = next(step for step in saved.steps if step.name == "base")
        reference = saved.steps[0]
        for step in saved.steps:
            if step.name in {"idle/0", "idle/5"}:
                assert step.method == "canonical"
                assert assets[step.output_asset_id] == b"canonical-base"
                provenance = json.loads(assets[step.provenance_asset_id])
                assert json.loads(provenance["meta"])["source_request_id"] == str(
                    base.request_id
                )
                assert step.source_asset_id == base.output_asset_id
            else:
                assert step.method == "generated"
                if step.name != reference.name:
                    assert step.source_asset_id == reference.output_asset_id
                    if sheet_mode:
                        assert "do not reproduce the sheet" in step.prompt
        assert all(step.strength is None for step in saved.steps)
        assert all(step.raw_asset_id in assets for step in saved.steps)
        assert all(step.provenance_asset_id in assets for step in saved.steps)
        expected_start = (
            ["character-sheet", "base", "idle/0"]
            if sheet_mode
            else ["base", "idle/0", "idle/1"]
        )
        assert [step.name for step in saved.steps[:3]] == expected_start
        assert saved.manifest.base_asset_id == base.output_asset_id
        assert assets[saved.manifest.atlas_asset_id] == b"webp-atlas"
        assert not any(call[2].endswith("/pet") for call in services.calls)

    asyncio.run(run())


def test_saved_generation_reuses_exact_persisted_output():
    async def run():
        pets, _, job = await setup_running()
        request_id = uuid4()
        generation = GeneratedImage(
            model_id=IMAGE_MODEL,
            seeds=[42],
            images_base64=["unused"],
            mime_type="image/png",
        )

        async def content(key, asset_id):
            if asset_id == uuid5(request_id, "raw"):
                return PNG
            assert asset_id == uuid5(request_id, "provenance")
            return generation.model_dump_json(exclude={"images_base64"}).encode()

        pets.content = content
        raw, metadata = await pets.saved_generation(job, request_id)
        assert raw == PNG
        assert metadata.seeds == [42]
        assert metadata.model_id == IMAGE_MODEL

    asyncio.run(run())


def test_sheet_recovery_reuses_saved_generation_before_starting_base():
    async def run():
        pets, services, job = await setup_running("character_sheet")
        request_id = uuid4()
        job = await pets.save(
            job,
            worker_id=pets.owner,
            pending_step_name="character-sheet",
            pending_request_id=request_id,
        )
        generation = GeneratedImage(
            model_id=IMAGE_MODEL,
            seeds=[42],
            images_base64=[base64.b64encode(PNG).decode()],
            mime_type="image/png",
        )
        pets.saved_generation = AsyncMock(return_value=(PNG, generation))
        pets.prepare_output = AsyncMock(return_value=b"saved-sheet")
        pets.upload = AsyncMock()
        pets.image = AsyncMock()
        await pets.generate_frame(job, None, -1, ["character-sheet", "base"], True, 1)
        pets.image.assert_not_awaited()
        pets.saved_generation.assert_awaited_once_with(job, request_id)
        pets.prepare_output.assert_awaited_once_with(job, PNG, None, -1)
        saved = services.jobs[job.id]
        assert [step.name for step in saved.steps] == ["character-sheet"]
        assert saved.steps[0].request_id == request_id
        assert saved.pending_request_id is None
        assert saved.reference_mode == "character_sheet" and saved.total_steps == 75
        assert pets.upload.await_count == 1
        assert pets.upload.call_args.args[3] == b"saved-sheet"

    asyncio.run(run())


def test_cross_agent_job_response_is_rejected():
    job = PetJob(
        id=PET_ID,
        agent_key="other",
        prompt="fox",
        version=1,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    with pytest.raises(ServiceError, match="invalid pet job"):
        PetJobs.checked(job.model_dump(), KEY, PET_ID)


def test_malformed_cleanup_response_does_not_claim_stopped():
    async def run():
        pets = PetJobs(
            SimpleNamespace(json_request=AsyncMock(return_value={"status": "queued"}))
        )
        with pytest.raises(ServiceError, match="did not confirm"):
            await pets.stop_request(uuid4())

    asyncio.run(run())


def test_frame_request_uses_untiled_reference_and_one_fixed_canvas():
    async def run():
        services = Storage()
        response = SimpleNamespace(
            is_success=True,
            json=lambda: {
                "next_batch_capacity": 5,
                "image": {
                    "seeds": [7],
                    "images_base64": [base64.b64encode(PNG).decode()],
                    "mime_type": "image/png",
                },
            },
        )
        services.client.post = AsyncMock(return_value=response)
        pets = PetJobs(services)
        result = await pets.image(
            uuid4(), "run right", PNG, IMAGE_MODEL, 42, batch_collection_target=3
        )
        assert result.next_batch_capacity == 5
        assert "next_batch_capacity" not in result.model_dump()
        request = services.client.post.call_args.kwargs["json"]
        assert request["mode"] == "reference"
        assert request["seed"] == 42
        assert request["batch_collection_target"] == 3
        assert (
            request["image"]
            == "data:image/png;base64," + base64.b64encode(PNG).decode()
        )
        assert (request["width"], request["height"]) == (
            SOURCE_CELL[0],
            SOURCE_CELL[1],
        )
        assert "strength" not in request and "mask" not in request

    asyncio.run(run())


@pytest.mark.parametrize("cancel_phase", ["during_request", "after_output"])
@pytest.mark.parametrize("reference_mode", ["single_image", "character_sheet"])
def test_cancelled_frame_resumes_without_regenerating_completed_frames(
    monkeypatch, cancel_phase, reference_mode
):
    async def inline_thread(function, *args):
        return function(*args)

    monkeypatch.setattr("kadan.runner.pet_jobs.asyncio.to_thread", inline_thread)
    monkeypatch.setattr("kadan.runner.pet_jobs.prepare_base", lambda raw: b"base")
    monkeypatch.setattr(
        "kadan.runner.pet_jobs.prepare_character_sheet", lambda raw: b"sheet"
    )
    monkeypatch.setattr(
        "kadan.runner.pet_jobs.prepare_frame", lambda raw, row, frame: PNG
    )
    monkeypatch.setattr("kadan.runner.pet_jobs.assemble_atlas", lambda frames: b"atlas")
    monkeypatch.setattr("kadan.runner.pet_jobs.validate_row", lambda row, frames: None)

    async def run():
        pets, services, job = await setup_running(reference_mode)
        extra = int(reference_mode == "character_sheet")
        assets = {}
        requests = []
        cancelled = False

        async def upload(job, asset_id, label, data, media_type):
            nonlocal cancelled
            assets[asset_id] = data
            if label == "idle/2" and cancel_phase == "after_output" and not cancelled:
                cancelled = True
                await pets.cancel(KEY, job.id)
                raise asyncio.CancelledError
            return asset_id

        async def content(key, asset_id):
            if asset_id not in assets:
                raise ServiceError("Missing", 404)
            return assets[asset_id]

        async def image(
            request_id,
            prompt,
            source,
            model_id,
            seed=None,
            *,
            batch_collection_target=None,
        ):
            nonlocal cancelled
            requests.append(request_id)
            if (
                len(requests) == 3 + extra
                and cancel_phase == "during_request"
                and not cancelled
            ):
                cancelled = True
                await pets.cancel(KEY, job.id)
                raise asyncio.CancelledError
            return GeneratedImage(
                model_id=IMAGE_MODEL,
                seeds=[42],
                images_base64=[base64.b64encode(PNG).decode()],
                mime_type="image/png",
            )

        pets.upload, pets.content, pets.image = upload, content, image
        await pets.process(job)
        stopped = services.jobs[job.id]
        assert stopped.status == "cancelled"
        expected = (["character-sheet"] if extra else []) + ["base", "idle/0", "idle/1"]
        assert [step.name for step in stopped.steps] == expected
        saved_steps = deepcopy(stopped.steps)
        saved_bytes = {
            step.output_asset_id: assets[step.output_asset_id] for step in saved_steps
        }
        resumed = await pets.resume(KEY, job.id)
        resumed = await pets.save(
            resumed,
            worker_id=pets.owner,
            status="running",
            owner_id=pets.owner,
            lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
        )
        await pets.process(resumed)
        finished = services.jobs[job.id]
        assert finished.status == "ready"
        assert finished.steps[: 3 + extra] == saved_steps
        assert finished.reference_mode == reference_mode
        assert all(assets[asset_id] == data for asset_id, data in saved_bytes.items())
        assert len(requests) == (73 if cancel_phase == "during_request" else 72) + extra
        if cancel_phase == "after_output":
            assert finished.steps[3 + extra].request_id == stopped.pending_request_id
        assert all(
            requests.count(step.request_id) == 1
            for step in saved_steps
            if step.method == "generated"
        )
        assert not any(call[2].endswith("/pet") for call in services.calls)

    asyncio.run(run())


def test_last_frame_is_validated_with_saved_row_before_commit(monkeypatch):
    async def inline_thread(function, *args):
        return function(*args)

    monkeypatch.setattr("kadan.runner.pet_jobs.asyncio.to_thread", inline_thread)

    async def run():
        pets = PetJobs(Storage())
        outputs = [uuid4() for _ in range(6)]
        job = SimpleNamespace(
            agent_key=KEY,
            steps=[
                SimpleNamespace(name=name, output_asset_id=asset)
                for name, asset in zip(
                    ["base", *[f"idle/{i}" for i in range(5)]], outputs, strict=True
                )
            ],
        )
        pets.content = AsyncMock(
            side_effect=[b"one", b"two", b"three", b"four", b"five"]
        )
        monkeypatch.setattr(
            "kadan.runner.pet_jobs.prepare_frame", lambda raw, row, frame: b"six"
        )
        seen = []

        def reject_repeated(row, frames):
            seen.append(frames)
            raise ValueError("repeated row")

        monkeypatch.setattr("kadan.runner.pet_jobs.validate_row", reject_repeated)
        with pytest.raises(ValueError, match="repeated row"):
            await pets.prepare_output(job, PNG, ROWS[0], 5)
        assert seen == [[b"one", b"two", b"three", b"four", b"five", PNG]]
        assert [call.args[1] for call in pets.content.call_args_list] == outputs[1:]

    asyncio.run(run())


def test_rejected_saved_frame_persists_new_seed_before_retry():
    async def run():
        pets, services, job = await setup_running()
        base = PetStep(
            name="base",
            output_asset_id=uuid4(),
            raw_asset_id=uuid4(),
            provenance_asset_id=uuid4(),
            prompt="fox",
            request_id=uuid4(),
            model_id=IMAGE_MODEL,
            seed=2**32 - 1,
        )
        neutral = base.model_copy(update={"name": "idle/0", "output_asset_id": uuid4()})
        old_request = uuid4()
        job = await pets.save(
            job,
            worker_id=pets.owner,
            steps=[base, neutral],
            completed_steps=2,
            pending_step_name="idle/1",
            pending_request_id=old_request,
            pending_seed=base.seed,
        )
        pets.content = AsyncMock(return_value=b"canonical-base")
        pets.saved_generation = AsyncMock(
            return_value=(
                PNG,
                GeneratedImage(
                    model_id=IMAGE_MODEL,
                    seeds=[base.seed],
                    images_base64=["unused"],
                    mime_type="image/png",
                ),
            )
        )
        pets.prepare_output = AsyncMock(side_effect=ValueError("boundary"))

        async def image(
            request_id, prompt, source, model_id, seed, *, batch_collection_target=None
        ):
            persisted = services.jobs[job.id]
            assert request_id != old_request
            assert persisted.pending_request_id == request_id
            assert seed == persisted.pending_seed == 0
            assert persisted.steps == [base, neutral]
            raise ServiceError("stop after observing submitted retry")

        pets.image = image
        with pytest.raises(ServiceError, match="stop after"):
            await pets.generate(job)

    asyncio.run(run())


def test_revision_regenerates_only_rejected_middle_frame(monkeypatch):
    async def inline_thread(function, *args):
        return function(*args)

    monkeypatch.setattr("kadan.runner.pet_jobs.asyncio.to_thread", inline_thread)
    monkeypatch.setattr("kadan.runner.pet_jobs.assemble_atlas", lambda frames: b"atlas")

    async def run():
        pets, services, job = await setup_running()
        names = [
            "base",
            *(f"{row.name}/{i}" for row in ROWS for i in range(row.frame_count)),
        ]
        previous = [
            PetStep(
                name=name,
                output_asset_id=uuid4(),
                raw_asset_id=uuid4(),
                provenance_asset_id=uuid4(),
                prompt="fox",
                request_id=uuid4(),
                model_id=IMAGE_MODEL,
                seed=7,
            )
            for name in names
        ]
        rejected = previous[3]
        retained = [step for step in previous if step != rejected]
        job = await pets.save(
            job,
            worker_id=pets.owner,
            steps=retained,
            completed_steps=73,
            rejections=[{"step": rejected.model_dump(), "reason": "wrong eyelids"}],
        )
        pets.content = AsyncMock(return_value=PNG)
        pets.prepare_output = AsyncMock(return_value=PNG)
        pets.validate_completed_row = AsyncMock()
        pets.upload = AsyncMock()
        pets.image = AsyncMock(
            return_value=GeneratedImage(
                model_id=IMAGE_MODEL,
                seeds=[8],
                images_base64=[base64.b64encode(PNG).decode()],
                mime_type="image/png",
            )
        )
        await pets.generate(job)
        saved = services.jobs[job.id]
        assert saved.status == "ready"
        assert [step.name for step in saved.steps] == names
        assert [step for step in saved.steps if step.name != rejected.name] == retained
        assert saved.rejections[0].step == rejected
        assert saved.steps[3].seed == 8
        pets.image.assert_awaited_once()
        assert pets.image.await_args.args[-1] == 8
        assert "wrong eyelids" in pets.image.await_args.args[1]

    asyncio.run(run())


def test_cohort_persists_out_of_order_successes_when_one_frame_fails(monkeypatch):
    async def run():
        pets, services, job = await setup_running()
        base = PetStep(
            name="base",
            output_asset_id=uuid4(),
            raw_asset_id=uuid4(),
            provenance_asset_id=uuid4(),
            prompt="owl",
            request_id=uuid4(),
            model_id=IMAGE_MODEL,
            seed=42,
        )
        neutral = base.model_copy(update={"name": "idle/0", "output_asset_id": uuid4()})
        job = await pets.owned_update(
            job,
            steps=[base, neutral],
            completed_steps=2,
            **pets.pending_state(
                [
                    PetPendingFrame(name=f"idle/{index}", request_id=uuid4(), seed=42)
                    for index in range(1, 5)
                ]
            ),
        )
        pets.saved_generation = AsyncMock(return_value=None)
        pets.content = AsyncMock(return_value=PNG)
        pets.prepare_output = AsyncMock(return_value=PNG)
        pets.validate_completed_row = AsyncMock()
        pets.upload = AsyncMock()
        started = asyncio.Event()
        checkpointed = asyncio.Event()
        last_started = asyncio.Event()
        original_frame = pets.generate_frame

        async def delayed_frame(
            job, row, frame, order, recover, batch_collection_target
        ):
            if frame == 4:
                await checkpointed.wait()
            return await original_frame(
                job, row, frame, order, recover, batch_collection_target
            )

        pets.generate_frame = delayed_frame
        requests = {}
        original_request = services.json_request

        async def track_checkpoint(*args, **kwargs):
            result = await original_request(*args, **kwargs)
            if [step.name for step in services.jobs[job.id].steps] == [
                "base",
                "idle/0",
                "idle/3",
            ]:
                checkpointed.set()
            return result

        services.json_request = track_checkpoint

        async def image(
            request_id,
            prompt,
            source,
            model_id,
            seed=None,
            *,
            batch_collection_target=None,
        ):
            pending = services.jobs[job.id].pending_images
            # The last sibling starts after another result was persisted. Its
            # immutable cohort hint must remain four even though pending shrank.
            assert batch_collection_target == 4
            assert len(pending) == (3 if checkpointed.is_set() else 4)
            frame = next(item for item in pending if item.request_id == request_id)
            assert frame.seed == seed == 42
            assert source == PNG and model_id == IMAGE_MODEL
            requests[frame.name] = asyncio.get_running_loop().create_future()
            if len(requests) == 3:
                started.set()
            if len(requests) == 4:
                last_started.set()
            return await requests[frame.name]

        pets.image = image
        task = asyncio.create_task(pets.process(job))
        await asyncio.wait_for(started.wait(), timeout=2)
        result = GeneratedImage(
            model_id=IMAGE_MODEL,
            seeds=[42],
            images_base64=[base64.b64encode(PNG).decode()],
            mime_type="image/png",
        )
        requests["idle/3"].set_result(result)
        await asyncio.wait_for(checkpointed.wait(), timeout=2)
        await asyncio.wait_for(last_started.wait(), timeout=2)
        assert not task.done()
        requests["idle/1"].set_exception(ServiceError("frame failed"))
        requests["idle/4"].set_result(result)
        requests["idle/2"].set_result(result)
        await asyncio.wait_for(task, timeout=2)
        failed = services.jobs[job.id]
        assert failed.status == "failed"
        assert [step.name for step in failed.steps] == [
            "base",
            "idle/0",
            "idle/2",
            "idle/3",
            "idle/4",
        ]
        assert [item.name for item in failed.pending_images] == ["idle/1"]
        assert failed.pending_seed == 42
        assert len(requests) == 4
        assert not any(call[2].endswith("/pet") for call in services.calls)

        saved = deepcopy(failed.steps)
        resumed = await pets.resume(KEY, job.id)
        assert resumed.steps == saved
        assert resumed.pending_images == failed.pending_images
        assert (
            services.calls[-2][2]
            == f"/v1/images/requests/{failed.pending_request_id}/cancel"
        )

    asyncio.run(run())


def test_cancel_cohort_fences_every_request_and_resume_preserves_pending_seeds(
    monkeypatch,
):
    monkeypatch.setattr("kadan.runner.pet_jobs.LEASE_RENEWAL_SECONDS", 0.01)

    async def run():
        pets, services, job = await setup_running()
        base = PetStep(
            name="base",
            output_asset_id=uuid4(),
            raw_asset_id=uuid4(),
            provenance_asset_id=uuid4(),
            prompt="owl",
            request_id=uuid4(),
            model_id=IMAGE_MODEL,
            seed=42,
        )
        neutral = base.model_copy(update={"name": "idle/0", "output_asset_id": uuid4()})
        job = await pets.owned_update(
            job,
            steps=[base, neutral],
            completed_steps=2,
            **pets.pending_state(
                [
                    PetPendingFrame(name=f"idle/{index}", request_id=uuid4(), seed=42)
                    for index in range(1, 5)
                ]
            ),
        )
        pets.saved_generation = AsyncMock(return_value=None)
        pets.content = AsyncMock(return_value=PNG)
        started = asyncio.Event()
        requests = []

        async def image(request_id, *args, **kwargs):
            requests.append(request_id)
            if len(requests) == 4:
                started.set()
            await asyncio.Future()

        pets.image = image
        task = asyncio.create_task(pets.process(job))
        await asyncio.wait_for(started.wait(), timeout=2)
        pending = deepcopy(services.jobs[job.id].pending_images)
        await PetClient(services).cancel(KEY, job.id)
        await asyncio.wait_for(task, timeout=2)
        cancelled = services.jobs[job.id]
        assert cancelled.status == "cancelled"
        assert cancelled.steps == [base, neutral]
        assert cancelled.pending_images == pending
        cancellations = [call[2] for call in services.calls if call[0] == "Hikari"]
        assert set(cancellations) == {
            f"/v1/images/requests/{request_id}/cancel" for request_id in requests
        }
        resumed = await pets.resume(KEY, job.id)
        assert resumed.status == "queued"
        assert resumed.steps == [base, neutral]
        assert resumed.pending_images == pending
        assert [frame.seed for frame in pending] == [42] * 4
        assert len([call for call in services.calls if call[0] == "Hikari"]) == 8

    asyncio.run(run())


def test_failed_cohort_cleanup_attempts_every_request_before_refusing_resume():
    async def run():
        pets, services, job = await setup_running()
        job = await pets.save(
            job,
            worker_id=pets.owner,
            status="failed",
            owner_id=None,
            lease_expires_at=None,
            pending_step_name="idle/1",
            pending_request_id=uuid4(),
            pending_seed=11,
            additional_pending_frames=[
                {"name": "idle/2", "request_id": uuid4(), "seed": 12},
                {"name": "idle/3", "request_id": uuid4(), "seed": 13},
            ],
        )
        pets.stop_request = AsyncMock(side_effect=[ServiceError("unknown"), None, None])
        with pytest.raises(ServiceError, match="unknown"):
            await pets.resume(KEY, job.id)
        assert pets.stop_request.await_count == 3
        assert services.jobs[job.id].status == "failed"
        assert services.jobs[job.id].pending_images == job.pending_images

    asyncio.run(run())


def test_producer_tracks_reference_capacity_and_pressure_without_base_hint(monkeypatch):
    async def inline_thread(function, *args):
        return function(*args)

    monkeypatch.setattr("kadan.runner.pet_jobs.asyncio.to_thread", inline_thread)
    rows = ROWS[1:]
    monkeypatch.setattr("kadan.runner.pet_jobs.ROWS", rows)
    monkeypatch.setattr("kadan.runner.pet_jobs.assemble_atlas", lambda frames: b"atlas")

    async def run():
        pets, services, job = await setup_running()
        pets.content = AsyncMock(return_value=PNG)
        pets.prepare_output = AsyncMock(return_value=PNG)
        pets.validate_completed_row = AsyncMock()
        pets.upload = AsyncMock()
        cohorts = []
        hints = [2, 3, 5, 2, None, 32, 32, 32]
        submitted = []

        async def image(
            request_id,
            prompt,
            source,
            model_id,
            seed=None,
            *,
            batch_collection_target=None,
        ):
            pending = services.jobs[job.id].pending_images
            assert request_id in {frame.request_id for frame in pending}
            if source is None:
                # A base-image capacity cannot size reference-frame cohorts.
                hint = 32
                assert batch_collection_target == 1
            else:
                members = tuple(frame.request_id for frame in pending)
                assert batch_collection_target == len(members)
                if members not in cohorts:
                    cohorts.append(members)
                hint = hints[cohorts.index(members)]
                assert seed == 42
                submitted.append(request_id)
            # Give every persisted sibling a chance to submit before completion.
            await asyncio.sleep(0)
            return GeneratedImage(
                model_id=IMAGE_MODEL,
                seeds=[42],
                images_base64=[base64.b64encode(PNG).decode()],
                mime_type="image/png",
                next_batch_capacity=hint,
            )

        pets.image = image
        await pets.generate(job)
        sizes = [len(cohort) for cohort in cohorts]
        assert sizes[:7] == [1, 2, 3, 5, 2, 1, 32]
        assert sizes[7:] == [sum(row.frame_count for row in rows) - sum(sizes[:7])]
        assert len(submitted) == len(set(submitted)) == sum(sizes)
        finished = services.jobs[job.id]
        assert finished.status == "ready"
        assert finished.completed_steps == 1 + sum(sizes)
        assert finished.pending_images == []
        assert all(step.seed == 42 for step in finished.steps)
        assert not any(call[2].endswith("/pet") for call in services.calls)
        for call in pets.upload.call_args_list:
            if call.args[-1] == "application/json" and call.args[2].endswith(
                "-provenance"
            ):
                assert "next_batch_capacity" not in json.loads(call.args[3])

    asyncio.run(run())


@pytest.mark.parametrize("capacity", [0, 33, -1])
def test_generated_image_rejects_unbounded_capacity_hint(capacity):
    with pytest.raises(ValueError):
        GeneratedImage(
            model_id=IMAGE_MODEL,
            seeds=[42],
            images_base64=[base64.b64encode(PNG).decode()],
            mime_type="image/png",
            next_batch_capacity=capacity,
        )


def test_initial_pet_artwork_explicitly_uses_local_generation():
    async def run():
        services = Storage()
        services.client.post = AsyncMock(
            return_value=SimpleNamespace(
                is_success=True,
                json=lambda: {
                    "image": {
                        "seeds": [7],
                        "images_base64": [base64.b64encode(PNG).decode()],
                        "mime_type": "image/png",
                    }
                },
            )
        )
        await PetJobs(services).image(uuid4(), "fox", None, IMAGE_MODEL)
        assert services.client.post.call_args.kwargs["json"]["provider"] == "local"

    asyncio.run(run())
