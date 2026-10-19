"""One durable image queue, independent of the conversation inference loop."""

import asyncio
import base64
import hashlib
import io
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import UUID, uuid4, uuid5

import httpx
from kadan_tracing import job_span, traced
from PIL import Image
from pydantic import BaseModel, Field

from ..api.config import ServiceError
from ..api.pydantic_models.agent_runs import ImageJob, ImageJobState
from ..api.pydantic_models.resources import AgentKey

logger = logging.getLogger(__name__)
LEASE_SECONDS = 60
HEARTBEAT_SECONDS = 15
MAX_IMAGE_BYTES = 20 * 1024 * 1024
TERMINAL = {"completed", "failed", "cancelled"}


class ImageAsset(BaseModel):
    id: UUID
    agent_key: AgentKey
    media_type: Literal["image/png"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    deleted_at: datetime | None = None


class GeneratedImage(BaseModel):
    images_base64: list[str] = Field(min_length=1, max_length=1)
    mime_type: Literal["image/png"] | None = None


def validate_png(data: bytes) -> None:
    if len(data) > MAX_IMAGE_BYTES or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ServiceError("Hikari returned invalid image data.", 502, "invalid_image")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or image.width * image.height > 16777216:
                raise ValueError("Invalid PNG dimensions")
            image.verify()
    except Exception as error:
        raise ServiceError(
            "Image data could not be verified.", 502, "invalid_image"
        ) from error


class ImageJobs:
    def __init__(self, services):
        self.services = services
        self.owner = uuid4()
        self.wakeup = asyncio.Event()
        self.lock = asyncio.Lock()
        self.runner: asyncio.Task | None = None
        self.stopping = False

    def start(self) -> None:
        if self.runner is None:
            self.stopping = False
            self.runner = asyncio.create_task(self.run(), name="image-jobs")

    async def close(self) -> None:
        self.stopping = True
        if self.runner is not None:
            self.runner.cancel()
            await asyncio.gather(self.runner, return_exceptions=True)
            self.runner = None

    async def get(self, key: str, job_id: UUID) -> ImageJob:
        job = ImageJob.model_validate(
            await self.services.json_request(
                "Tsuchi", "GET", f"/api/agents/{key}/image-jobs/{job_id}"
            )
        )
        if job.agent_key != key or job.id != job_id:
            raise ServiceError(
                "Tsuchi returned the wrong image job.", 502, "invalid_response"
            )
        return job

    async def save(self, job: ImageJob, **changes: Any) -> ImageJob:
        state = ImageJobState.model_validate(
            {**job.model_dump(include=set(ImageJobState.model_fields)), **changes}
        )
        value = await self.services.json_request(
            "Tsuchi",
            "PUT",
            f"/api/agents/{job.agent_key}/image-jobs/{job.id}",
            json={
                **state.model_dump(mode="json"),
                "expected_version": job.version,
                "worker_id": str(self.owner),
            },
        )
        saved = ImageJob.model_validate(value)
        if saved.status in TERMINAL:
            self.services.runs.wakeup.set()
        return saved

    async def run(self) -> None:
        while not self.stopping:
            self.wakeup.clear()
            try:
                values = await self.services.json_request(
                    "Tsuchi", "GET", "/api/image-jobs", params={"active": "true"}
                )
                for value in values:
                    job = ImageJob.model_validate(value)
                    recovering = job.status in {"running", "cancelling"}
                    if recovering and job.lease_expires_at > datetime.now(UTC):
                        continue
                    if job.status not in {"queued", "running", "cancelling"}:
                        continue
                    with job_span(
                        "kadan.image_jobs.process",
                        job,
                        queued_at=job.updated_at if not recovering else None,
                        recovering=recovering,
                    ):
                        try:
                            job = await self.save(
                                job,
                                status="cancelling"
                                if job.status == "cancelling"
                                else "running",
                                owner_id=self.owner,
                                lease_expires_at=datetime.now(UTC)
                                + timedelta(seconds=LEASE_SECONDS),
                            )
                        except ServiceError as error:
                            if error.status == 409:
                                continue
                            raise
                        await self.process(job, recovering=recovering)
                    self.wakeup.set()
                    break
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Image queue could not reconcile persisted jobs")
            try:
                await asyncio.wait_for(self.wakeup.wait(), timeout=2)
            except TimeoutError:
                pass

    async def owned(self, job: ImageJob) -> ImageJob:
        current = await self.get(job.agent_key, job.id)
        if current.owner_id != self.owner or current.status not in {
            "running",
            "cancelling",
        }:
            raise ServiceError("Image job ownership changed.", 409, "image_lease_lost")
        return current

    async def heartbeat(self, job: ImageJob, work: asyncio.Task) -> None:
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_SECONDS)
                async with self.lock:
                    current = await self.owned(job)
                    if current.status == "cancelling":
                        work.cancel()
                        return
                    await self.save(
                        current,
                        lease_expires_at=datetime.now(UTC)
                        + timedelta(seconds=LEASE_SECONDS),
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Image job lease renewal failed; stopping image work")
            work.cancel()

    async def stop_request(self, job: ImageJob) -> None:
        result = await self.services.json_request(
            "Hikari", "POST", f"/v1/images/requests/{job.request_id}/cancel"
        )
        if not isinstance(result, dict) or result.get("status") not in {
            "cancelled",
            "not_active",
        }:
            raise ServiceError(
                "Hikari did not confirm image cleanup.", 502, "invalid_response"
            )

    async def process(self, job: ImageJob, *, recovering: bool = False) -> None:
        # Keep work separate from the queue task so cancellation does not stop the queue.
        work = asyncio.create_task(
            self.recover(job) if recovering else self.generate(job)
        )
        heartbeat = asyncio.create_task(self.heartbeat(job, work))
        try:
            await work
        except (Exception, asyncio.CancelledError) as error:  # noqa: BLE001 - every worker exit must fence its request.
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)
            try:
                # Fence even after ownership changes. Recovery never resubmits this
                # request, so a stale worker may safely stop its own inference.
                await self.stop_request(job)
                async with self.lock:
                    current = await self.owned(job)
                    if current.status == "cancelling":
                        await self.finish(current, "cancelled")
                    else:
                        asset_id = await self.saved_asset(current)
                        if asset_id is not None:
                            await self.finish(current, "completed", asset_id=asset_id)
                        else:
                            await self.finish(
                                current,
                                "failed",
                                error=str(error)[:10000]
                                or "Image generation was interrupted before an image was saved.",
                            )
            except Exception:
                # A failed fence/lease write remains active for expiry recovery.
                logger.exception("Image job cleanup was not confirmed")
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

    @traced("kadan.image_jobs.finish")
    async def finish(self, job: ImageJob, status: str, **changes: Any) -> ImageJob:
        return await self.save(
            job, status=status, owner_id=None, lease_expires_at=None, **changes
        )

    @traced("kadan.image_jobs.recover")
    async def recover(self, job: ImageJob) -> None:
        # A generation with an uncertain outcome must never be silently resubmitted.
        await self.stop_request(job)
        async with self.lock:
            current = await self.owned(job)
            if current.status == "cancelling":
                await self.finish(current, "cancelled")
                return
            asset_id = await self.saved_asset(current)
            if asset_id is not None:
                await self.finish(current, "completed", asset_id=asset_id)
            else:
                await self.finish(
                    current,
                    "failed",
                    error="Image generation was interrupted before an image was saved; it was not resubmitted.",
                )

    @traced("kadan.image_jobs.saved_asset")
    async def saved_asset(self, job: ImageJob) -> UUID | None:
        asset_id = uuid5(job.id, "image")
        try:
            value = await self.services.json_request(
                "Tsuchi", "GET", f"/api/agents/{job.agent_key}/assets/{asset_id}"
            )
        except ServiceError as error:
            if error.status == 404:
                return None
            raise
        asset = ImageAsset.model_validate(value)
        if asset.id != asset_id or asset.agent_key != job.agent_key or asset.deleted_at:
            raise ServiceError(
                "Saved image metadata is invalid.", 502, "invalid_image_asset"
            )
        response = await self.services.client.get(
            self.services.settings.tsuchi_url
            + f"/api/agents/{job.agent_key}/assets/{asset_id}/content",
            timeout=60,
        )
        response.raise_for_status()
        validate_png(response.content)
        if hashlib.sha256(response.content).hexdigest() != asset.sha256:
            raise ServiceError(
                "Saved image checksum does not match.", 502, "invalid_image_asset"
            )
        return asset_id

    @traced("kadan.image_jobs.generate")
    async def generate(self, job: ImageJob) -> None:
        response = await self.services.client.post(
            self.services.settings.hikari_url + "/v1/images/generations",
            json={"prompt": job.prompt, "count": 1, "aspect": "1:1"},
            headers={"x-kadan-request-id": str(job.request_id)},
            timeout=httpx.Timeout(None, connect=3),
        )
        if not response.is_success:
            raise ServiceError(
                f"Hikari image generation failed (HTTP {response.status_code}).",
                502,
                "image_generation_failed",
            )
        result = GeneratedImage.model_validate(response.json()["image"])
        if len(result.images_base64[0]) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
            raise ServiceError("Generated image is too large.", 502, "invalid_image")
        raw = base64.b64decode(result.images_base64[0], validate=True)
        validate_png(raw)
        asset_id = uuid5(job.id, "image")
        async with self.lock:
            current = await self.owned(job)
            if current.status == "cancelling":
                raise asyncio.CancelledError
        value = await self.services.json_request(
            "Tsuchi",
            "POST",
            f"/api/agents/{job.agent_key}/assets",
            files={"file": ("generated-image.png", raw, "image/png")},
            data={
                "title": "Generated image",
                "tags": json.dumps(["generated-image", str(job.id)]),
                "asset_id": str(asset_id),
            },
        )
        asset = ImageAsset.model_validate(value)
        if (
            asset.id != asset_id
            or asset.agent_key != job.agent_key
            or asset.deleted_at
            or asset.sha256 != hashlib.sha256(raw).hexdigest()
        ):
            raise ServiceError(
                "Tsuchi did not confirm the saved image.", 502, "invalid_image_asset"
            )
        async with self.lock:
            current = await self.owned(job)
            if current.status == "cancelling":
                raise asyncio.CancelledError
            await self.finish(current, "completed", asset_id=asset_id)
