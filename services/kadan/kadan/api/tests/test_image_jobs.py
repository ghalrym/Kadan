import asyncio
import base64
import hashlib
import io
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4, uuid5

import httpx
import pytest
from PIL import Image

from kadan.api.config import ServiceError
from kadan.api.pydantic_models.agent_runs import ImageJob
from kadan.runner.image_jobs import ImageJobs


def png():
    output = io.BytesIO()
    Image.new("RGB", (2, 2)).save(output, format="PNG")
    return output.getvalue()


class Storage:
    def __init__(self):
        self.assets = {}
        self.fences = []
        self.fail_fence = False
        self.completed = asyncio.Event()
        self.runs = SimpleNamespace(wakeup=asyncio.Event())
        self.settings = SimpleNamespace(
            hikari_url="http://hikari", tsuchi_url="http://tsuchi"
        )
        self.client = SimpleNamespace(
            post=AsyncMock(
                return_value=httpx.Response(
                    200,
                    json={
                        "image": {"images_base64": [base64.b64encode(png()).decode()]}
                    },
                )
            ),
            get=AsyncMock(
                return_value=httpx.Response(
                    200, content=png(), request=httpx.Request("GET", "http://tsuchi")
                )
            ),
        )

    async def json_request(self, service, method, path, **kwargs):
        if service == "Hikari":
            self.fences.append(path)
            if self.fail_fence:
                raise ServiceError("No cleanup acknowledgement")
            return {"status": "not_active"}
        if "/assets" in path:
            if method == "GET":
                asset = self.assets.get(path.rsplit("/", 1)[1])
                if asset is None:
                    raise ServiceError("Missing", 404)
                return asset
            data = kwargs["data"]
            raw = kwargs["files"]["file"][1]
            asset = {
                "id": data["asset_id"],
                "agent_key": self.job.agent_key,
                "media_type": "image/png",
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            self.assets[data["asset_id"]] = asset
            return asset
        if method == "PUT":
            body = kwargs["json"]
            if body["expected_version"] != self.job.version:
                raise ServiceError("Conflict", 409)
            state = {
                k: v
                for k, v in body.items()
                if k not in {"expected_version", "worker_id"}
            }
            self.job = ImageJob.model_validate(
                {**self.job.model_dump(), **state, "version": self.job.version + 1}
            )
            if self.job.status in {"completed", "failed", "cancelled"}:
                self.completed.set()
        return self.job.model_dump(mode="json")


def setup():
    services = Storage()
    images = ImageJobs(services)
    job_id = uuid4()
    services.job = ImageJob(
        id=job_id,
        agent_key="andrew",
        run_id=uuid4(),
        conversation_id=uuid4(),
        prompt="An owl",
        request_id=uuid4(),
        event_run_id=uuid5(job_id, "event"),
        status="running",
        owner_id=images.owner,
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
        version=1,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    return services, images, services.job


def test_success_saves_image_and_wakes_conversation_without_changing_model():
    async def run():
        services, images, job = setup()
        await images.process(job)
        assert services.job.status == "completed"
        assert services.job.asset_id == uuid5(job.id, "image")
        assert services.job.owner_id is None
        assert services.runs.wakeup.is_set()
        body = services.client.post.call_args.kwargs
        assert body["json"] == {"prompt": job.prompt, "count": 1, "aspect": "1:1"}
        assert body["headers"] == {"x-kadan-request-id": str(job.request_id)}
        assert services.fences == []

    asyncio.run(run())


def test_failed_generation_emits_terminal_failure_after_fencing():
    async def run():
        services, images, job = setup()
        services.client.post.return_value = httpx.Response(503)
        await images.process(job)
        assert services.job.status == "failed"
        assert "503" in services.job.error
        assert services.fences == [f"/v1/images/requests/{job.request_id}/cancel"]
        assert services.runs.wakeup.is_set()

    asyncio.run(run())


@pytest.mark.parametrize("saved", [True, False])
def test_restart_reconciles_saved_asset_and_never_resubmits(saved):
    async def run():
        services, images, job = setup()
        if saved:
            asset_id = uuid5(job.id, "image")
            services.assets[str(asset_id)] = {
                "id": str(asset_id),
                "agent_key": job.agent_key,
                "media_type": "image/png",
                "sha256": hashlib.sha256(png()).hexdigest(),
            }
        await images.process(job, recovering=True)
        services.client.post.assert_not_called()
        assert services.fences
        assert services.job.status == ("completed" if saved else "failed")
        if not saved:
            assert "not resubmitted" in services.job.error

    asyncio.run(run())


def test_recovered_cancellation_waits_for_fence_then_emits_cancelled():
    async def run():
        services, images, job = setup()
        services.job = job.model_copy(update={"status": "cancelling"})
        await images.process(services.job, recovering=True)
        services.client.post.assert_not_called()
        assert services.job.status == "cancelled"
        assert services.fences
        assert services.runs.wakeup.is_set()

    asyncio.run(run())


def test_missing_cancellation_ack_retains_recoverable_lease():
    async def run():
        services, images, job = setup()
        services.job = job.model_copy(update={"status": "cancelling"})
        services.fail_fence = True
        await images.process(services.job, recovering=True)
        assert services.job.status == "cancelling"
        assert services.job.owner_id == images.owner
        assert not services.runs.wakeup.is_set()

    asyncio.run(run())


def test_cancellation_during_generation_does_not_deliver_image():
    async def run():
        services, images, job = setup()
        response = services.client.post.return_value

        async def cancel_during_generation(*args, **kwargs):
            services.job = services.job.model_copy(
                update={"status": "cancelling", "version": 2}
            )
            return response

        services.client.post.side_effect = cancel_during_generation
        await images.process(job)
        assert services.job.status == "cancelled"
        assert services.assets == {}
        assert services.fences

    asyncio.run(run())


def test_corrupt_png_never_becomes_completed():
    async def run():
        services, images, job = setup()
        services.client.post.return_value = httpx.Response(
            200,
            json={
                "image": {
                    "images_base64": [
                        base64.b64encode(b"\x89PNG\r\n\x1a\ninvalid").decode()
                    ]
                }
            },
        )
        await images.process(job)
        assert services.job.status == "failed"
        assert services.assets == {}

    asyncio.run(run())


def test_ambiguous_upload_recovers_deterministic_saved_image():
    async def run():
        services, images, job = setup()
        original = services.json_request

        async def uncertain_reply(service, method, path, **kwargs):
            result = await original(service, method, path, **kwargs)
            if method == "POST" and path.endswith("/assets"):
                raise ServiceError("Connection lost after upload")
            return result

        services.json_request = uncertain_reply
        await images.process(job)
        assert services.job.status == "completed"
        assert services.job.asset_id == uuid5(job.id, "image")
        services.client.post.assert_awaited_once()

    asyncio.run(run())


def test_shutdown_fences_generation_and_marks_interrupted_failure():
    async def run():
        services, images, job = setup()
        started = asyncio.Event()

        async def pending(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()

        services.client.post.side_effect = pending
        work = asyncio.create_task(images.process(job))
        await started.wait()
        work.cancel()
        await work
        assert services.job.status == "failed"
        assert "interrupted" in services.job.error
        assert services.fences

    asyncio.run(run())


def test_lease_loss_fences_request_without_overwriting_new_owner(monkeypatch):
    async def run():
        services, images, job = setup()
        new_owner = uuid4()

        async def lost_lease(*args, **kwargs):
            services.job = services.job.model_copy(
                update={"owner_id": new_owner, "version": 2}
            )
            await asyncio.Event().wait()

        services.client.post.side_effect = lost_lease
        await asyncio.wait_for(images.process(job), timeout=1)
        assert services.job.owner_id == new_owner
        assert services.job.status == "running"
        assert services.fences
        assert not services.runs.wakeup.is_set()

    monkeypatch.setattr("kadan.runner.image_jobs.HEARTBEAT_SECONDS", 0.001)
    asyncio.run(run())


def test_heartbeat_observes_cancellation_of_pending_generation(monkeypatch):
    async def run():
        services, images, job = setup()

        async def cancel_pending(*args, **kwargs):
            services.job = services.job.model_copy(
                update={"status": "cancelling", "version": 2}
            )
            await asyncio.Event().wait()

        services.client.post.side_effect = cancel_pending
        await asyncio.wait_for(images.process(job), timeout=1)
        assert services.job.status == "cancelled"
        assert services.fences
        assert services.runs.wakeup.is_set()

    monkeypatch.setattr("kadan.runner.image_jobs.HEARTBEAT_SECONDS", 0.001)
    asyncio.run(run())
