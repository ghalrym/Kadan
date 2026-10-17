import asyncio
import base64
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4, uuid5

import pytest

from kadan.api.builtin_mcp import BuiltinTools
from kadan.api.config import ServiceError
from kadan.api.pet_artwork import ROWS, SOURCE_CELL
from kadan.api.pet_jobs import IMAGE_MODEL, PetJobs
from kadan.api.pydantic_models.pets import GeneratedImage, PetJob

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


async def setup_running():
    services = Storage()
    pets = PetJobs(services)
    job = await pets.create(KEY, PET_ID, "An ink-painted owl")
    job = await pets.save(
        job,
        worker=True,
        status="running",
        owner_id=pets.owner,
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
    )
    return pets, services, job


def test_create_persists_idempotent_job_without_inference_or_selection():
    async def run():
        services = Storage()
        pets = PetJobs(services)
        one = await pets.create(KEY, PET_ID, "A fox")
        two = await pets.create(KEY, PET_ID, "A fox")
        assert one.id == two.id == PET_ID
        assert one.status == "queued"
        assert all(
            call[0] == "Tsuchi" and call[2].endswith("/pets") for call in services.calls
        )
        assert pets.wakeup.is_set()

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
            worker=True,
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
            worker=True,
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
            job, worker=True, pending_step_name="base", pending_request_id=uuid4()
        )
        await pets.cancel(KEY, job.id)
        pets.generate = AsyncMock(side_effect=asyncio.CancelledError)
        pets.stop_request = AsyncMock(side_effect=ServiceError("cleanup unknown"))
        await pets.process(job)
        assert services.jobs[job.id].status == "cancelling"

    asyncio.run(run())


def test_pipeline_conditions_every_pose_on_same_base_and_never_selects(monkeypatch):
    async def inline_thread(function, *args):
        return function(*args)

    monkeypatch.setattr("kadan.api.pet_jobs.asyncio.to_thread", inline_thread)

    async def run():
        pets, services, job = await setup_running()
        assets = {}
        sources = []

        async def upload(job, asset_id, label, data, media_type):
            assets[asset_id] = data
            return asset_id

        async def content(key, asset_id):
            return assets[asset_id]

        async def image(request_id, prompt, source, model_id):
            sources.append(source)
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
            "kadan.api.pet_jobs.prepare_base", lambda raw: b"canonical-base"
        )
        monkeypatch.setattr(
            "kadan.api.pet_jobs.prepare_frame", lambda raw, row, frame: PNG
        )
        monkeypatch.setattr(
            "kadan.api.pet_jobs.assemble_atlas", lambda strips: b"webp-atlas"
        )
        monkeypatch.setattr("kadan.api.pet_jobs.validate_row", lambda row, frames: None)
        await asyncio.wait_for(pets.generate(job), timeout=5)
        saved = services.jobs[job.id]
        assert saved.status == "ready"
        assert saved.completed_steps == 74
        assert len(saved.steps) == 74
        assert len(sources) == 74 and sources[0] is None
        assert all(source == b"canonical-base" for source in sources[1:])
        assert all(step.strength is None for step in saved.steps)
        assert all(step.raw_asset_id in assets for step in saved.steps)
        assert all(step.provenance_asset_id in assets for step in saved.steps)
        assert [step.name for step in saved.steps[:3]] == ["base", "idle/0", "idle/1"]
        assert saved.manifest.base_asset_id == saved.steps[0].output_asset_id
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


@pytest.mark.parametrize("action", ["create", "status", "cancel", "resume"])
def test_pet_tools_call_only_the_bound_agent_operation(action):
    async def run():
        pets, services, job = await setup_running()
        services.pets = pets
        method = "get" if action == "status" else action
        operation = AsyncMock(return_value=job)
        setattr(pets, method, operation)
        tool = BuiltinTools(services, KEY)
        result = await tool.pet(
            action, PET_ID, "An owl" if action == "create" else None
        )
        if action == "create":
            operation.assert_awaited_once_with(KEY, PET_ID, "An owl")
        else:
            operation.assert_awaited_once_with(KEY, PET_ID)
        assert result["status"] == "running"
        assert "Only the user" in result["next_action"]

    asyncio.run(run())


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
                "image": {
                    "seeds": [7],
                    "images_base64": [base64.b64encode(PNG).decode()],
                    "mime_type": "image/png",
                }
            },
        )
        services.client.post = AsyncMock(return_value=response)
        pets = PetJobs(services)
        await pets.image(uuid4(), "run right", PNG, IMAGE_MODEL)
        request = services.client.post.call_args.kwargs["json"]
        assert request["mode"] == "reference"
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
def test_cancelled_frame_resumes_without_regenerating_completed_frames(
    monkeypatch, cancel_phase
):
    async def inline_thread(function, *args):
        return function(*args)

    monkeypatch.setattr("kadan.api.pet_jobs.asyncio.to_thread", inline_thread)
    monkeypatch.setattr("kadan.api.pet_jobs.prepare_base", lambda raw: b"base")
    monkeypatch.setattr("kadan.api.pet_jobs.prepare_frame", lambda raw, row, frame: PNG)
    monkeypatch.setattr("kadan.api.pet_jobs.assemble_atlas", lambda frames: b"atlas")
    monkeypatch.setattr("kadan.api.pet_jobs.validate_row", lambda row, frames: None)

    async def run():
        pets, services, job = await setup_running()
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

        async def image(request_id, prompt, source, model_id):
            nonlocal cancelled
            requests.append(request_id)
            if (
                len(requests) == 4
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
        assert [step.name for step in stopped.steps] == ["base", "idle/0", "idle/1"]
        saved_steps = deepcopy(stopped.steps)
        saved_bytes = {
            step.output_asset_id: assets[step.output_asset_id] for step in saved_steps
        }
        resumed = await pets.resume(KEY, job.id)
        resumed = await pets.save(
            resumed,
            worker=True,
            status="running",
            owner_id=pets.owner,
            lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
        )
        await pets.process(resumed)
        finished = services.jobs[job.id]
        assert finished.status == "ready"
        assert finished.steps[:3] == saved_steps
        assert all(assets[asset_id] == data for asset_id, data in saved_bytes.items())
        assert len(requests) == (75 if cancel_phase == "during_request" else 74)
        if cancel_phase == "after_output":
            assert finished.steps[3].request_id == stopped.pending_request_id
        assert all(requests.count(step.request_id) == 1 for step in saved_steps)
        assert not any(call[2].endswith("/pet") for call in services.calls)

    asyncio.run(run())


def test_last_frame_is_validated_with_saved_row_before_commit(monkeypatch):
    async def inline_thread(function, *args):
        return function(*args)

    monkeypatch.setattr("kadan.api.pet_jobs.asyncio.to_thread", inline_thread)

    async def run():
        pets = PetJobs(Storage())
        outputs = [uuid4() for _ in range(6)]
        job = SimpleNamespace(
            agent_key=KEY,
            steps=[SimpleNamespace(output_asset_id=asset) for asset in outputs],
        )
        pets.content = AsyncMock(
            side_effect=[b"one", b"two", b"three", b"four", b"five"]
        )
        monkeypatch.setattr(
            "kadan.api.pet_jobs.prepare_frame", lambda raw, row, frame: b"six"
        )
        seen = []

        def reject_repeated(row, frames):
            seen.append(frames)
            raise ValueError("repeated row")

        monkeypatch.setattr("kadan.api.pet_jobs.validate_row", reject_repeated)
        with pytest.raises(ValueError, match="repeated row"):
            await pets.prepare_output(job, PNG, ROWS[0], 5)
        assert seen == [[b"one", b"two", b"three", b"four", b"five", b"six"]]
        assert [call.args[1] for call in pets.content.call_args_list] == outputs[1:]

    asyncio.run(run())
