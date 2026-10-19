"""Pet management and persistence shared by API callers and durable workers."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid5

from pydantic import TypeAdapter, ValidationError

from .config import ServiceError
from .pet_artwork import ROWS
from .pydantic_models.pets import PetAsset, PetJob, PetManifest, PetState
from .pydantic_models.resources import AgentKey

if TYPE_CHECKING:
    from .services import LocalServices

TERMINAL = {"ready", "failed", "cancelled", "interrupted"}


class PetClient:
    def __init__(self, services: LocalServices):
        self.services = services
        self.lock = asyncio.Lock()

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
        return job

    async def import_atlas(
        self, key: str, pet_id: UUID, prompt: str, atlas: bytes, sprite_version: int
    ) -> PetJob:
        # Asset and job IDs derive from the caller's UUID, so a retry is idempotent.
        atlas_id = await self.upload_asset(
            key, pet_id, uuid5(pet_id, "atlas"), "atlas", atlas, "image/webp"
        )
        manifest = PetManifest(
            sprite_version=sprite_version,
            rows=11 if sprite_version == 2 else 9,
            atlas_asset_id=atlas_id,
            animations=[
                {"name": row.name, "row": row.index, "durations": list(row.durations)}
                for row in ROWS[:9]
            ],
        )
        value = await self.services.json_request(
            "Tsuchi",
            "POST",
            self.path(key) + "/imports",
            json={
                "id": str(pet_id),
                "prompt": prompt,
                "manifest": manifest.model_dump(mode="json"),
            },
        )
        return self.checked(value, key, pet_id)

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
        self, job: PetJob, *, worker_id: UUID | None = None, **changes: Any
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
                "worker_id": str(worker_id) if worker_id else None,
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
            return job

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

    async def upload_asset(
        self,
        key: str,
        pet_id: UUID,
        asset_id: UUID,
        label: str,
        data: bytes,
        media_type: str,
    ) -> UUID:
        suffix = {"image/png": "png", "image/webp": "webp", "application/json": "json"}[
            media_type
        ]
        value = await self.services.json_request(
            "Tsuchi",
            "POST",
            f"/api/agents/{key}/assets",
            files={"file": (f"{label.replace('/', '-')}.{suffix}", data, media_type)},
            data={
                "title": f"Pet {label}",
                "tags": json.dumps(["pet", str(pet_id)]),
                "asset_id": str(asset_id),
            },
        )
        asset = PetAsset.model_validate(value)
        if (
            asset.id != asset_id
            or asset.agent_key != key
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
