"""Prove a fenced native cohort idle before its coupled replacement."""

import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from api.services.inference_errors import InferenceFailure
from kadan_tracing.rollout import Admission
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

WORKER_KINDS = frozenset(
    {"llm", "decision", "tts", "speech", "formatting", "image", "video"}
)


class NativeWorker(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    instance: str = Field(min_length=1)
    generation: UUID


class ColdStart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: Literal["cold_start"]
    generation: UUID
    old_workers: list[NativeWorker]

    @model_validator(mode="after")
    def complete_cohort(self):
        if (
            len(self.old_workers) != len(WORKER_KINDS)
            or {worker.kind for worker in self.old_workers} != WORKER_KINDS
            or len({worker.generation for worker in self.old_workers})
            != len(WORKER_KINDS)
        ):
            raise ValueError("Cold start requires a complete prior native cohort")
        return self


class NativeLane(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    worker: str
    busy: bool
    queued: int = Field(ge=0)
    releasing: bool


class NativeBarrier:
    def __init__(
        self,
        hub,
        gate: Admission,
        active: Callable[[], int],
        marker: Path | None = None,
    ):
        self.hub = hub
        self.gate = gate
        self.active = active
        self.api_generation = str(uuid4())
        self.started_fenced = gate.fenced
        self.cold_start: ColdStart | None = None
        if self.started_fenced:
            try:
                cold_start = ColdStart.model_validate_json(
                    (marker or gate.fence.with_name("native.json")).read_text()
                )
                if os.getenv("KADAN_ROLLOUT_GENERATION") == str(cold_start.generation):
                    self.cold_start = cold_start
            except (OSError, ValidationError):
                pass
        self.cold_start_proven = False
        self.proof: dict | None = None
        self.proof_epoch: int | None = None

    def cohort(self):
        workers = tuple(
            sorted(self.hub.workers.values(), key=lambda worker: worker.kind)
        )
        if (
            len(workers) != len(WORKER_KINDS)
            or {worker.kind for worker in workers} != WORKER_KINDS
            or len({worker.generation for worker in workers}) != len(WORKER_KINDS)
            or not all(worker.ready.is_set() for worker in workers)
            or any(
                worker.preparing is not None and not worker.preparing.done()
                for worker in workers
            )
        ):
            raise InferenceFailure(
                "Native rollout requires the complete ready worker cohort.", 503
            )
        if not self.hub.rollout_safe:
            raise InferenceFailure("Native connection history is uncertain.", 503)
        if self.started_fenced:
            if self.cold_start is None:
                raise InferenceFailure(
                    "Fenced API startup requires a cold-start attestation.", 503
                )
            previous = {
                str(worker.generation) for worker in self.cold_start.old_workers
            }
            if any(
                worker.generation in previous
                or worker.rollout_generation != str(self.cold_start.generation)
                for worker in workers
            ):
                raise InferenceFailure(
                    "A prior native generation survived the cold start.", 503
                )
        return workers

    def ready(self):
        try:
            self.cohort()
        except InferenceFailure:
            return False
        return not self.started_fenced or self.cold_start_proven

    def proof_valid(self):
        return (
            self.proof is not None
            and self.proof_epoch == self.hub.rollout_epoch
            and self.hub.rollout_frozen
            and self.gate.fenced
            and self.active() == 0
            and self.hub.rollout_safe
        )

    def status(self):
        return {
            "api_generation": self.api_generation,
            "cold_start_generation": str(self.cold_start.generation)
            if self.cold_start
            else None,
            "native_idle": self.proof_valid(),
            "workers": self.proof["workers"] if self.proof_valid() else [],
        }

    async def prove(self):
        if not self.gate.fenced or self.active():
            raise InferenceFailure(
                "Native barrier requires fenced and drained API admission.", 409
            )
        async with self.hub.maintenance:
            if not self.gate.fenced or self.active():
                raise InferenceFailure(
                    "API work changed before the native barrier.", 409
                )
            workers = self.cohort()
            epoch = self.hub.rollout_epoch
            # Maintenance and all later submissions stay frozen through replacement.
            self.hub.rollout_frozen = True
            self.proof = None
            if any(worker.controls for worker in workers):
                raise InferenceFailure("Native controls are still in flight.", 409)
            replies = await asyncio.gather(
                *(worker.control("status", timeout=5) for worker in workers)
            )
            for worker, reply in zip(workers, replies, strict=True):
                try:
                    lanes = [
                        NativeLane.model_validate(value) for value in reply["workers"]
                    ]
                except (KeyError, TypeError, ValidationError) as error:
                    raise InferenceFailure(
                        "Native status did not establish idle lanes.", 503
                    ) from error
                if (
                    len(lanes) != 2
                    or {lane.worker for lane in lanes} != {"store", worker.kind}
                    or any(lane.busy or lane.queued or lane.releasing for lane in lanes)
                ):
                    raise InferenceFailure(
                        "Native store or inference lane is not idle.", 409
                    )
            if (
                not self.gate.fenced
                or self.active()
                or epoch != self.hub.rollout_epoch
                or workers != self.cohort()
            ):
                raise InferenceFailure(
                    "Native cohort changed during its idle proof.", 409
                )
            self.cold_start_proven = True
            self.proof_epoch = epoch
            self.proof = {
                "native_idle": True,
                "api_generation": self.api_generation,
                "cold_start_generation": str(self.cold_start.generation)
                if self.cold_start
                else None,
                "workers": [
                    NativeWorker(
                        kind=worker.kind,
                        instance=worker.instance,
                        generation=worker.generation,
                    ).model_dump(mode="json")
                    for worker in workers
                ],
            }
            return self.proof
