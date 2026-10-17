"""Durable, agent-scoped pet creation. HTTP and MCP calls only enqueue work."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4, uuid5

import httpx
from pydantic import TypeAdapter, ValidationError

from .config import ServiceError
from .pet_artwork import (
    ROWS,
    SOURCE_CELL,
    PetRow,
    assemble_atlas,
    base_prompt,
    frame_name,
    frame_prompt,
    prepare_base,
    prepare_frame,
    validate_row,
)
from .pydantic_models.pets import (
    GeneratedImage,
    PetAsset,
    PetJob,
    PetManifest,
    PetState,
    PetStep,
)
from .pydantic_models.resources import AgentKey

if TYPE_CHECKING:
    from .services import LocalServices

logger = logging.getLogger(__name__)
LEASE_SECONDS = 60
MAX_IMAGE_BYTES = 20 * 1024**2
IMAGE_MODEL = "qwen-image-2.1-turbo"
TERMINAL = {"ready", "failed", "cancelled", "interrupted"}


class PetJobs:
    def __init__(self, services: LocalServices):
        self.services = services
        self.owner = uuid4()
        self.wakeup = asyncio.Event()
        self.runner: asyncio.Task | None = None
        self.active: asyncio.Task | None = None
        self.active_id: UUID | None = None
        self.stopping = False
        self.lock = asyncio.Lock()

    def start(self) -> None:
        self.runner = asyncio.create_task(self.run(), name="pet-queue")

    async def close(self) -> None:
        self.stopping = True
        if self.runner:
            self.runner.cancel()
            await asyncio.gather(self.runner, return_exceptions=True)

    @staticmethod
    def path(key: str, pet_id: UUID | None = None) -> str:
        TypeAdapter(AgentKey).validate_python(key, strict=True)
        return f"/api/agents/{key}/pets" + (f"/{pet_id}" if pet_id else "")

    @staticmethod
    def checked(
        value: Any, key: str | None = None, pet_id: UUID | None = None
    ) -> PetJob:
        try:
            job = PetJob.model_validate(value)
            if (key is not None and job.agent_key != key) or (
                pet_id is not None and job.id != pet_id
            ):
                raise ValueError("Wrong pet namespace or ID")
            return job
        except (ValueError, ValidationError) as error:
            raise ServiceError(
                "Tsuchi returned an invalid pet job.", 502, "invalid_response"
            ) from error

    async def get(self, key: str, pet_id: UUID) -> PetJob:
        return self.checked(
            await self.services.json_request("Tsuchi", "GET", self.path(key, pet_id)),
            key,
            pet_id,
        )

    async def list(self, key: str) -> list[PetJob]:
        values = await self.services.json_request("Tsuchi", "GET", self.path(key))
        return [self.checked(value, key) for value in values]

    async def create(self, key: str, pet_id: UUID, prompt: str) -> PetJob:
        value = await self.services.json_request(
            "Tsuchi", "POST", self.path(key), json={"id": str(pet_id), "prompt": prompt}
        )
        job = self.checked(value, key, pet_id)
        self.wakeup.set()
        return job

    async def selected(self, key: str) -> PetJob | None:
        self.path(key)
        value = await self.services.json_request(
            "Tsuchi", "GET", f"/api/agents/{key}/pet"
        )
        return self.checked(value, key) if value is not None else None

    async def select(self, key: str, pet_id: UUID | None) -> PetJob | None:
        self.path(key)
        value = await self.services.json_request(
            "Tsuchi",
            "PUT",
            f"/api/agents/{key}/pet",
            json={"pet_id": str(pet_id) if pet_id else None},
        )
        return self.checked(value, key, pet_id) if value is not None else None

    async def save(
        self, job: PetJob, *, worker: bool = False, **changes: Any
    ) -> PetJob:
        state = PetState.model_validate(job.model_dump()).model_dump(mode="json")
        state.update(changes)
        # Revalidate/serialize UUIDs and dates without including immutable job fields.
        state = PetState.model_validate(state).model_dump(mode="json")
        value = await self.services.json_request(
            "Tsuchi",
            "PUT",
            self.path(job.agent_key, job.id),
            json={
                **state,
                "expected_version": job.version,
                "worker_id": str(self.owner) if worker else None,
            },
        )
        return self.checked(value, job.agent_key, job.id)

    async def cancel(self, key: str, pet_id: UUID) -> PetJob:
        async with self.lock:
            job = await self.get(key, pet_id)
            if job.status in TERMINAL:
                return job
            status = "cancelled" if job.status == "queued" else "cancelling"
            job = await self.save(
                job, status=status, stage=status, cancel_requested=True
            )
            if self.active_id == pet_id and self.active:
                self.active.cancel()
            self.wakeup.set()
            return job

    async def resume(self, key: str, pet_id: UUID) -> PetJob:
        async with self.lock:
            job = await self.get(key, pet_id)
            if job.status not in {"failed", "cancelled", "interrupted"}:
                raise ServiceError(
                    "Only a stopped pet job can be resumed.", 409, "pet_not_stopped"
                )
            # The old request is fenced before any new inference can be submitted.
            if job.pending_request_id:
                await self.stop_request(job.pending_request_id)
            job = await self.save(
                job, status="queued", stage="queued", cancel_requested=False, error=None
            )
            self.wakeup.set()
            return job

    async def run(self) -> None:
        while not self.stopping:
            self.wakeup.clear()
            try:
                values = await self.services.json_request(
                    "Tsuchi", "GET", "/api/pet-jobs"
                )
                for value in values:
                    job = self.checked(value)
                    if job.status in {"running", "cancelling"}:
                        if (
                            job.lease_expires_at
                            and job.lease_expires_at <= datetime.now(UTC)
                        ):
                            if job.pending_request_id:
                                await self.stop_request(job.pending_request_id)
                            await self.save(
                                job,
                                status="cancelled"
                                if job.cancel_requested
                                else "interrupted",
                                stage="cancelled"
                                if job.cancel_requested
                                else "interrupted",
                                owner_id=None,
                                lease_expires_at=None,
                                error="The worker stopped. Saved artwork is retained; resume to continue."
                                if not job.cancel_requested
                                else None,
                            )
                        continue
                    if job.status == "queued":
                        try:
                            job = await self.save(
                                job,
                                worker=True,
                                status="running",
                                stage="preparing",
                                owner_id=self.owner,
                                lease_expires_at=datetime.now(UTC)
                                + timedelta(seconds=LEASE_SECONDS),
                            )
                        except ServiceError as error:
                            if error.status == 409:
                                continue
                            raise
                        self.active_id = job.id
                        self.active = asyncio.create_task(
                            self.process(job), name=f"pet-{job.id}"
                        )
                        try:
                            await self.active
                        finally:
                            self.active = None
                            self.active_id = None
                        self.wakeup.set()
                        break
            except asyncio.CancelledError:
                raise
            except Exception:
                # Storage outages must not kill the durable queue. No inference runs
                # until a lease has been successfully persisted.
                logger.exception("Pet queue could not reconcile persisted jobs")
            try:
                await asyncio.wait_for(self.wakeup.wait(), timeout=5)
            except TimeoutError:
                pass

    async def owned_update(self, job: PetJob, **changes: Any) -> PetJob:
        async with self.lock:
            current = await self.get(job.agent_key, job.id)
            if (
                current.owner_id != self.owner
                or current.status != "running"
                or current.cancel_requested
            ):
                raise ServiceError(
                    "Pet ownership changed or cancellation was requested.",
                    409,
                    "pet_lease_lost",
                )
            return await self.save(current, worker=True, **changes)

    async def heartbeat(self, job: PetJob, work: asyncio.Task) -> None:
        try:
            while True:
                await asyncio.sleep(10)
                await self.owned_update(
                    job,
                    lease_expires_at=datetime.now(UTC)
                    + timedelta(seconds=LEASE_SECONDS),
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Pet lease could not be renewed; stopping inference")
            work.cancel()

    async def process(self, job: PetJob) -> None:
        work = asyncio.current_task()
        heartbeat = asyncio.create_task(self.heartbeat(job, work))
        try:
            await self.generate(job)
        except (Exception, asyncio.CancelledError) as error:  # noqa: BLE001 - every worker exit must release its durable lease.
            try:
                async with self.lock:
                    current = await self.get(job.agent_key, job.id)
                    if current.owner_id == self.owner:
                        if current.pending_request_id:
                            await self.stop_request(current.pending_request_id)
                        cancelled = current.cancel_requested
                        status = (
                            "cancelled"
                            if cancelled
                            else (
                                "interrupted"
                                if isinstance(error, asyncio.CancelledError)
                                else "failed"
                            )
                        )
                        await self.save(
                            current,
                            worker=True,
                            status=status,
                            stage=status,
                            owner_id=None,
                            lease_expires_at=None,
                            error=None
                            if cancelled
                            else str(error)[:10000]
                            or "The worker stopped. Resume to continue from saved artwork.",
                        )
            except Exception:
                # Keep the running/cancelling lease visible; recovery will reconcile
                # after expiry instead of claiming cancellation or lost output saved.
                logger.exception("Pet cleanup was not confirmed")
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

    async def stop_request(self, request_id: UUID) -> None:
        result = await self.services.json_request(
            "Hikari", "POST", f"/v1/images/requests/{request_id}/cancel"
        )
        if not isinstance(result, dict) or result.get("status") not in {
            "cancelled",
            "not_active",
        }:
            raise ServiceError(
                "Hikari did not confirm image cleanup.", 502, "invalid_response"
            )

    async def content(self, key: str, asset_id: UUID) -> bytes:
        value = await self.services.json_request(
            "Tsuchi", "GET", f"/api/agents/{key}/assets/{asset_id}"
        )
        asset = PetAsset.model_validate(value)
        if asset.id != asset_id or asset.agent_key != key or asset.deleted_at:
            raise ServiceError(
                "Pet artwork is missing or in trash.", 409, "invalid_pet_asset"
            )
        response = await self.services.client.get(
            self.services.settings.tsuchi_url
            + f"/api/agents/{key}/assets/{asset_id}/content",
            timeout=60,
        )
        response.raise_for_status()
        if (
            len(response.content) > MAX_IMAGE_BYTES
            or hashlib.sha256(response.content).hexdigest() != asset.sha256
        ):
            raise ServiceError(
                "Tsuchi returned invalid pet artwork.", 502, "invalid_pet_asset"
            )
        return response.content

    async def upload(
        self, job: PetJob, asset_id: UUID, label: str, data: bytes, media_type: str
    ) -> UUID:
        suffix = {"image/png": "png", "image/webp": "webp", "application/json": "json"}[
            media_type
        ]
        value = await self.services.json_request(
            "Tsuchi",
            "POST",
            f"/api/agents/{job.agent_key}/assets",
            files={"file": (f"{label.replace('/', '-')}.{suffix}", data, media_type)},
            data={
                "title": f"Pet {label}",
                "tags": json.dumps(["pet", str(job.id)]),
                "asset_id": str(asset_id),
            },
        )
        asset = PetAsset.model_validate(value)
        if (
            asset.id != asset_id
            or asset.agent_key != job.agent_key
            or asset.deleted_at
            or asset.sha256 != hashlib.sha256(data).hexdigest()
            or asset.media_type != media_type
        ):
            raise ServiceError(
                "Tsuchi did not confirm the saved pet artwork.",
                502,
                "invalid_pet_asset",
            )
        return asset.id

    async def image(
        self,
        request_id: UUID,
        prompt: str,
        source: bytes | None,
        model_id: str | None,
        seed: int | None = None,
    ) -> GeneratedImage:
        body: dict[str, Any] = {
            "prompt": prompt,
            "count": 1,
            "aspect": "1:1",
            "model_id": IMAGE_MODEL,
        }
        if model_id:
            body["model_id"] = model_id
        if seed is not None:
            body["seed"] = seed
        if source is not None:
            body.update(
                image="data:image/png;base64," + base64.b64encode(source).decode(),
                mode="reference",
                width=SOURCE_CELL[0],
                height=SOURCE_CELL[1],
            )
        response = await self.services.client.post(
            self.services.settings.hikari_url
            + ("/v1/images/edits" if source is not None else "/v1/images/generations"),
            json=body,
            headers={"x-kadan-request-id": str(request_id)},
            timeout=httpx.Timeout(1800, connect=3),
        )
        if not response.is_success:
            raise ServiceError(
                f"Hikari could not generate this pet stage (HTTP {response.status_code}).",
                response.status_code,
                "pet_generation_failed",
            )
        result = GeneratedImage.model_validate(
            {**response.json()["image"], "model_id": body["model_id"]}
        )
        raw = base64.b64decode(result.images_base64[0], validate=True)
        if not raw.startswith(b"\x89PNG\r\n\x1a\n") or len(raw) > MAX_IMAGE_BYTES:
            raise ServiceError(
                "Hikari returned invalid pet artwork.", 502, "invalid_image"
            )
        return result

    async def saved_generation(
        self, job: PetJob, request_id: UUID
    ) -> tuple[bytes, GeneratedImage] | None:
        try:
            raw = await self.content(job.agent_key, uuid5(request_id, "raw"))
            metadata = json.loads(
                await self.content(job.agent_key, uuid5(request_id, "provenance"))
            )
            metadata["images_base64"] = [base64.b64encode(raw).decode()]
            return raw, GeneratedImage.model_validate(metadata)
        except ServiceError as error:
            if error.status != 404:
                raise
            return None

    async def prepare_output(
        self, job: PetJob, raw: bytes, row: PetRow | None, frame: int
    ) -> bytes:
        if row is None:
            return await asyncio.to_thread(prepare_base, raw)
        output = await asyncio.to_thread(prepare_frame, raw, row, frame)
        if frame == row.frame_count - 1:
            previous = [
                await self.content(job.agent_key, step.output_asset_id)
                for step in job.steps[-frame:]
            ]
            await asyncio.to_thread(validate_row, row, [*previous, output])
        return output

    async def generate(self, job: PetJob) -> None:
        frames = [(row, frame) for row in ROWS for frame in range(row.frame_count)]
        for index, item in enumerate([None, *frames]):
            if index < len(job.steps):
                continue
            row, frame = item if item is not None else (None, 0)
            name = frame_name(row, frame) if row is not None else "base"
            prompt = (
                frame_prompt(job.prompt, row, frame)
                if row is not None
                else base_prompt(job.prompt)
            )
            source_id = job.steps[0].output_asset_id if row is not None else None
            source = (
                await self.content(job.agent_key, source_id)
                if source_id is not None
                else None
            )
            seed = (
                job.pending_seed
                if job.pending_request_id
                else job.steps[0].seed
                if row is not None
                else None
            )
            recovered = (
                await self.saved_generation(job, job.pending_request_id)
                if job.pending_request_id
                else None
            )
            if recovered:
                try:
                    output = await self.prepare_output(job, recovered[0], row, frame)
                except ValueError:
                    # Keep initial poses on shared noise, but let a rejected frame
                    # change on retry. Persist this choice before submitting it.
                    seed = (recovered[1].seeds[0] + 1) % 2**32
                    recovered = None
            request_id = job.pending_request_id if recovered else uuid4()
            job = await self.owned_update(
                job,
                stage=(
                    f"Generating {row.name} frame {frame + 1}/{row.frame_count}"
                    if row is not None
                    else "Generating base"
                ),
                pending_step_name=name,
                pending_request_id=request_id,
                pending_seed=seed,
            )
            if recovered:
                raw, generation = recovered
            else:
                job = await self.owned_update(
                    job,
                    lease_expires_at=datetime.now(UTC)
                    + timedelta(seconds=LEASE_SECONDS),
                )
                generation = await self.image(
                    request_id,
                    prompt,
                    source,
                    job.steps[0].model_id if job.steps else None,
                    seed,
                )
                raw = base64.b64decode(generation.images_base64[0], validate=True)
                metadata = generation.model_dump(exclude={"images_base64"})
                await self.upload(
                    job,
                    uuid5(request_id, "provenance"),
                    f"{name}-provenance",
                    json.dumps(metadata).encode(),
                    "application/json",
                )
                await self.upload(
                    job, uuid5(request_id, "raw"), f"{name}-raw", raw, "image/png"
                )
            output_id = uuid5(request_id, "output")
            if not recovered:
                output = await self.prepare_output(job, raw, row, frame)
            await self.upload(job, output_id, name, output, "image/png")
            step = PetStep(
                name=name,
                source_asset_id=source_id if row else uuid5(request_id, "raw"),
                output_asset_id=output_id,
                raw_asset_id=uuid5(request_id, "raw"),
                provenance_asset_id=uuid5(request_id, "provenance"),
                prompt=prompt,
                request_id=request_id,
                model_id=generation.model_id,
                seed=generation.seeds[0],
                meta=generation.meta,
            )
            job = await self.owned_update(
                job,
                steps=[*job.steps, step],
                completed_steps=index + 1,
                pending_step_name=None,
                pending_request_id=None,
                pending_seed=None,
            )
        job = await self.owned_update(job, stage="Validating and assembling animation")
        frames = {
            step.name: await self.content(job.agent_key, step.output_asset_id)
            for step in job.steps[1:]
        }
        atlas = await asyncio.to_thread(assemble_atlas, frames)
        atlas_id = uuid5(job.id, "atlas")
        await self.upload(job, atlas_id, "atlas", atlas, "image/webp")
        manifest = PetManifest(
            atlas_asset_id=atlas_id,
            base_asset_id=job.steps[0].output_asset_id,
            animations=[
                {"name": row.name, "row": row.index, "durations": list(row.durations)}
                for row in ROWS[:9]
            ],
        )
        await self.owned_update(
            job,
            status="ready",
            stage="Ready to preview",
            manifest=manifest,
            owner_id=None,
            lease_expires_at=None,
            pending_step_name=None,
            pending_request_id=None,
            pending_seed=None,
        )
