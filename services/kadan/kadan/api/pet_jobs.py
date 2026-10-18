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
    PetPendingFrame,
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
MAX_PENDING_FRAMES = 32
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

    async def revise(
        self,
        key: str,
        pet_id: UUID,
        revision_id: UUID,
        expected_version: int,
        frame_names: list[str],
        reason: str,
    ) -> PetJob:
        value = await self.services.json_request(
            "Tsuchi",
            "POST",
            self.path(key, pet_id) + "/revisions",
            json={
                "id": str(revision_id),
                "expected_version": expected_version,
                "frame_names": frame_names,
                "reason": reason,
            },
        )
        job = self.checked(value, key, revision_id)
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
            # All old requests are fenced before any new inference can be submitted.
            await self.stop_pending(job)
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
            return await self.save(current, worker=True, **changes)

    @staticmethod
    def pending_state(frames: list[PetPendingFrame]) -> dict[str, Any]:
        primary = frames[0] if frames else None
        return {
            "pending_step_name": primary.name if primary else None,
            "pending_request_id": primary.request_id if primary else None,
            "pending_seed": primary.seed if primary else None,
            "additional_pending_frames": frames[1:],
        }

    async def stop_pending(self, job: PetJob) -> None:
        # Attempt every cancellation even when one confirmation fails. Keep the
        # durable cohort intact until all requests are confirmed stopped.
        results = await asyncio.gather(
            *(self.stop_request(frame.request_id) for frame in job.pending_images),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

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

    async def generate_frame(
        self,
        job: PetJob,
        row: PetRow | None,
        frame: int,
        order: list[str],
        recover: bool,
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
                worker=True,
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
                worker=True,
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
            tasks = [
                asyncio.create_task(
                    self.generate_frame(
                        job,
                        row,
                        frame,
                        order,
                        (frame_name(row, frame) if row is not None else "base")
                        in previous,
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
