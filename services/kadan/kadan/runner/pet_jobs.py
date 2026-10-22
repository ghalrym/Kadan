"""Durable, agent-scoped pet artwork worker."""

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
from kadan_tracing import job_span, traced

from ..api.config import ServiceError
from ..api.pet_artwork import (
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
from ..api.pet_client import PetClient
from ..api.pet_doctor import PetDoctor
from ..api.pydantic_models.pets import (
    GeneratedImage,
    PetAsset,
    PetJob,
    PetManifest,
    PetPendingFrame,
    PetStep,
)

if TYPE_CHECKING:
    from ..api.services import LocalServices

logger = logging.getLogger(__name__)
LEASE_SECONDS = 60
LEASE_RENEWAL_SECONDS = 10
MAX_IMAGE_BYTES = 20 * 1024**2
IMAGE_MODEL = "qwen-image-2.1-turbo"
MAX_PENDING_FRAMES = 32


class PetJobs(PetClient):
    def __init__(self, services: LocalServices):
        super().__init__(services)
        self.owner = uuid4()
        self.wakeup = asyncio.Event()
        self.runner: asyncio.Task | None = None
        self.active: asyncio.Task | None = None
        self.stopping = False

    def start(self) -> None:
        self.runner = asyncio.create_task(self.run(), name="pet-queue")

    async def close(self) -> None:
        self.stopping = True
        if self.runner:
            self.runner.cancel()
            await asyncio.gather(self.runner, return_exceptions=True)

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
                            await self.stop_pending(job)
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
                        with job_span(
                            "kadan.pet_jobs.process",
                            job,
                            queued_at=job.updated_at,
                            recovering=False,
                        ):
                            try:
                                job = await self.save(
                                    job,
                                    worker_id=self.owner,
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
                            self.active = asyncio.create_task(
                                self.process(job), name=f"pet-{job.id}"
                            )
                            try:
                                await self.active
                            finally:
                                self.active = None
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

    async def owned_job(self, job: PetJob) -> PetJob:
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
        return current

    async def owned_update(self, job: PetJob, **changes: Any) -> PetJob:
        async with self.lock:
            current = await self.owned_job(job)
            return await self.save(current, worker_id=self.owner, **changes)

    @staticmethod
    def pending_state(frames: list[PetPendingFrame]) -> dict[str, Any]:
        primary = frames[0] if frames else None
        return {
            "pending_step_name": primary.name if primary else None,
            "pending_request_id": primary.request_id if primary else None,
            "pending_seed": primary.seed if primary else None,
            "additional_pending_frames": frames[1:],
        }

    async def heartbeat(self, job: PetJob, work: asyncio.Task) -> None:
        try:
            while True:
                await asyncio.sleep(LEASE_RENEWAL_SECONDS)
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
            if job.doctor:
                await self.normalize(job)
            else:
                await self.generate(job)
        except (Exception, asyncio.CancelledError) as error:  # noqa: BLE001 - every worker exit must release its durable lease.
            try:
                async with self.lock:
                    current = await self.get(job.agent_key, job.id)
                    if current.owner_id == self.owner:
                        await self.stop_pending(current)
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
                            worker_id=self.owner,
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
        return await self.upload_asset(
            job.agent_key, job.id, asset_id, label, data, media_type
        )

    async def image(
        self,
        request_id: UUID,
        prompt: str,
        source: bytes | None,
        model_id: str | None,
        seed: int | None = None,
        *,
        batch_collection_target: int | None = None,
    ) -> GeneratedImage:
        body: dict[str, Any] = {
            "prompt": prompt,
            "count": 1,
            "aspect": "1:1",
            "model_id": IMAGE_MODEL,
        }
        if batch_collection_target is not None:
            body["batch_collection_target"] = batch_collection_target
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
        payload = response.json()
        result = GeneratedImage.model_validate(
            {
                **payload["image"],
                "model_id": body["model_id"],
                "next_batch_capacity": payload.get("next_batch_capacity"),
            }
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
        if row.name == "idle" and frame in {0, 5}:
            # These poses are the canonical neutral artwork, validated but unaltered.
            output = raw
        await self.validate_completed_row(job, output, row, frame)
        return output

    async def validate_completed_row(
        self, job: PetJob, output: bytes, row: PetRow | None, frame: int
    ) -> None:
        if row is None:
            return
        saved = {step.name: step for step in job.steps}
        names = [frame_name(row, index) for index in range(row.frame_count)]
        if all(name in saved for name in names if name != frame_name(row, frame)):
            complete = [
                output
                if index == frame
                else await self.content(job.agent_key, saved[name].output_asset_id)
                for index, name in enumerate(names)
            ]
            await asyncio.to_thread(validate_row, row, complete)

    @traced("kadan.pet_jobs.generate_frame")
    async def generate_frame(
        self,
        job: PetJob,
        row: PetRow | None,
        frame: int,
        order: list[str],
        recover: bool,
        batch_collection_target: int,
    ) -> int | None:
        name = frame_name(row, frame) if row is not None else "base"
        canonical = name in {"idle/0", "idle/5"}
        prompt = (
            "Reuse the canonical neutral artwork without altering its pose."
            if canonical
            else frame_prompt(job.prompt, row, frame)
            if row is not None
            else base_prompt(job.prompt)
        )
        source_id = job.steps[0].output_asset_id if row is not None else None
        source = (
            await self.content(job.agent_key, source_id)
            if source_id is not None
            else None
        )
        pending = next(item for item in job.pending_images if item.name == name)
        seed = pending.seed
        rejected = next(
            (entry for entry in reversed(job.rejections) if entry.step.name == name),
            None,
        )
        if rejected and not canonical:
            prompt += f" Correction requested for this frame: {rejected.reason}"
        output = None
        recovered = (
            await self.saved_generation(job, pending.request_id)
            if recover and not canonical
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
        request_id = uuid4() if recover and not recovered else pending.request_id
        async with self.lock:
            current = await self.owned_job(job)
            pending = PetPendingFrame(name=name, request_id=request_id, seed=seed)
            job = await self.save(
                current,
                worker_id=self.owner,
                lease_expires_at=datetime.now(UTC) + timedelta(seconds=LEASE_SECONDS),
                **self.pending_state(
                    [
                        pending if item.name == name else item
                        for item in current.pending_images
                    ]
                ),
            )
        if recovered:
            raw, generation = recovered
        else:
            if canonical:
                generation = GeneratedImage(
                    model_id=job.steps[0].model_id,
                    seeds=[job.steps[0].seed or 0],
                    images_base64=[base64.b64encode(source).decode()],
                    mime_type="image/png",
                    meta=json.dumps(
                        {
                            "method": "canonical",
                            "source_asset_id": str(source_id),
                            "source_request_id": str(job.steps[0].request_id),
                        }
                    ),
                )
            else:
                generation = await self.image(
                    request_id,
                    prompt,
                    source,
                    job.steps[0].model_id if job.steps else None,
                    seed,
                    batch_collection_target=batch_collection_target,
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
        step = PetStep(
            name=name,
            method="canonical" if canonical else "generated",
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
        if output is None:
            output = await self.prepare_output(job, raw, row, frame)
        await self.upload(job, output_id, name, output, "image/png")
        # Serialize row validation and checkpoint writes, so an out-of-order
        # completion cannot replace another frame or miss the final row check.
        async with self.lock:
            current = await self.owned_job(job)
            await self.validate_completed_row(current, output, row, frame)
            await self.save(
                current,
                worker_id=self.owner,
                steps=sorted(
                    [*current.steps, step], key=lambda saved: order.index(saved.name)
                ),
                completed_steps=len(current.steps) + 1,
                **self.pending_state(
                    [item for item in current.pending_images if item.name != name]
                ),
            )

        return (
            generation.next_batch_capacity
            if row is not None and not canonical
            else None
        )

    @traced("kadan.pet_jobs.generate")
    async def generate(self, job: PetJob) -> None:
        frames = [(row, frame) for row in ROWS for frame in range(row.frame_count)]
        items = [(None, 0), *frames]
        order = ["base", *(frame_name(row, frame) for row, frame in frames)]
        # Base generation has a different canvas and no reference conditioning.
        # Earn a reference-frame measurement before growing that producer window.
        capacity = 1
        while True:
            job = await self.owned_job(job)
            completed = {step.name for step in job.steps}
            remaining = [
                item
                for item, name in zip(items, order, strict=True)
                if name not in completed
            ]
            if not remaining:
                break
            previous = {item.name: item for item in job.pending_images}
            if previous:
                # Recovery must retain every fenced request, even when the new
                # worker has not yet learned the old cohort's capacity.
                remaining = [
                    (row, frame)
                    for row, frame in remaining
                    if (frame_name(row, frame) if row is not None else "base")
                    in previous
                ]
            cohort_limit = len(previous) if previous else capacity
            cohort = []
            for row, frame in remaining:
                name = frame_name(row, frame) if row is not None else "base"
                independent = name not in {"base", "idle/0", "idle/5"}
                if cohort and not independent:
                    break
                cohort.append((row, frame))
                if not independent or len(cohort) == cohort_limit:
                    break
            pending = []
            for row, frame in cohort:
                name = frame_name(row, frame) if row is not None else "base"
                seed = job.steps[0].seed if row is not None else None
                rejection = next(
                    (
                        entry
                        for entry in reversed(job.rejections)
                        if entry.step.name == name
                    ),
                    None,
                )
                if rejection and name not in {"idle/0", "idle/5"}:
                    seed = ((rejection.step.seed or 0) + 1) % 2**32
                pending.append(
                    previous.get(name)
                    or PetPendingFrame(
                        name=name,
                        request_id=uuid4(),
                        seed=seed,
                    )
                )
            # Persist every member before any HTTP request is launched. Recovery
            # can then fence all siblings even if this process disappears.
            first_row, first_frame = cohort[0]
            job = await self.owned_update(
                job,
                stage=(
                    f"Generating {first_row.name} frame {first_frame + 1}/{first_row.frame_count}"
                    if first_row is not None
                    else "Generating base"
                ),
                **self.pending_state(pending),
            )
            cohort_size = len(cohort)
            tasks = [
                asyncio.create_task(
                    self.generate_frame(
                        job,
                        row,
                        frame,
                        order,
                        (frame_name(row, frame) if row is not None else "base")
                        in previous,
                        cohort_size,
                    )
                )
                for row, frame in cohort
            ]
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            # A failed frame does not discard successful siblings. Stop before
            # adding a new cohort; resume retries only the unfinished frames.
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            if first_row is not None and frame_name(first_row, first_frame) not in {
                "idle/0",
                "idle/5",
            }:
                # All members share the reference canvas. Use the conservative
                # hint if pressure changed between sub-batches; missing advice
                # returns to a singleton instead of guessing from GPU count.
                capacity = min(MAX_PENDING_FRAMES, *(result or 1 for result in results))
        await self.assemble(job)

    async def normalize(self, job: PetJob) -> None:
        reference = await self.content(job.agent_key, job.steps[0].output_asset_id)
        doctor = await asyncio.to_thread(PetDoctor, reference)
        sources = {entry.step.name: entry.step for entry in job.rejections}
        for row in ROWS:
            for index in range(row.frame_count):
                job = await self.owned_job(job)
                name = frame_name(row, index)
                if any(step.name == name for step in job.steps):
                    continue
                job = await self.owned_update(job, stage=f"Doctor · {name}")
                original = sources[name]
                png = await self.content(job.agent_key, original.output_asset_id)
                output, result = await asyncio.to_thread(doctor.correct, png)
                step = original.model_copy(update={"normalization": result})
                if result.outcome == "corrected":
                    output_id = uuid5(job.id, name + "/normalized")
                    provenance_id = uuid5(job.id, name + "/normalization")
                    metadata = json.dumps(
                        {
                            "method": "normalized",
                            "source_step": original.model_dump(
                                mode="json", exclude={"meta"}
                            ),
                            "normalization": result.model_dump(),
                        },
                        sort_keys=True,
                    )
                    await self.upload(job, output_id, name, output, "image/png")
                    await self.upload(
                        job,
                        provenance_id,
                        name + "-normalization",
                        metadata.encode(),
                        "application/json",
                    )
                    step = step.model_copy(
                        update={
                            "method": "normalized",
                            "source_asset_id": original.output_asset_id,
                            "output_asset_id": output_id,
                            "provenance_asset_id": provenance_id,
                            "request_id": uuid5(job.id, name),
                            "meta": metadata,
                        }
                    )
                job = await self.owned_update(
                    job,
                    steps=[*job.steps, step],
                    completed_steps=len(job.steps) + 1,
                )
        await self.assemble(job)

    async def assemble(self, job: PetJob) -> None:
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
            additional_pending_frames=[],
        )
