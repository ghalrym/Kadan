"""Worker protocol models and outstanding jobs."""

import asyncio
import json
from enum import StrEnum
from typing import Literal

from api.pydantic_models.inference import (
    ImageProgress,
    InferenceProgress,
)
from api.pydantic_models.inference_task import InferenceTask
from api.pydantic_models.model_store import ModelStoreStatus
from api.services.inference_errors import InferenceFailure
from pydantic import BaseModel, ConfigDict, JsonValue, field_validator


def encode_request(message: dict) -> bytes:
    encoded = json.dumps(message, allow_nan=False, ensure_ascii=False).encode()
    task = message.get("input", {})
    maximum = (
        64 if task.get("feature") == "image" and task.get("image") is not None else 3
    )
    if len(encoded) > maximum * 1024 * 1024:
        raise InferenceFailure("Inference request exceeds its transport limit.", 413)
    return encoded


class JobState(StrEnum):
    QUEUED = "queued"
    PREPARING = "preparing"
    LOADING = "loading"
    RUNNING = "running"
    WAITING = "waiting_for_resources"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


TERMINAL = ("succeeded", "failed", "cancelled", "rejected")


class WorkerMemory(BaseModel):
    capacity: list[int]
    used: list[int]
    residents: int


class WorkerEvent(BaseModel):
    """A job event from a machine; control replies are handled separately."""

    model_config = ConfigDict(extra="forbid")
    id: str = ""
    state: Literal[
        "queued",
        "preparing",
        "loading",
        "running",
        "waiting_for_resources",
        "progress",
        "content",
        "succeeded",
        "failed",
        "cancelled",
        "rejected",
    ]
    trace_unix_ns: int | None = None
    worker: str | None = None
    result: JsonValue = None
    error: str = ""
    memory: WorkerMemory | None = None
    phase: str | None = None
    value: int = 0
    total: int = 0
    content: str | None = None
    model: str | None = None
    context: dict[str, int | None] | None = None
    status_code: int = 502
    models: ModelStoreStatus | None = None

    @field_validator("trace_unix_ns", mode="before")
    @classmethod
    def optional_trace_timestamp(cls, value: object) -> int | None:
        # Optional telemetry must not reject an otherwise valid completion.
        return value if type(value) is int and 0 < value < 2**63 else None


class InferenceJob:
    """An outstanding response and bounded streaming channel, never an executor."""

    def __init__(self, request: InferenceTask, streaming: bool):
        self.tracing = None
        self.request = request
        self.sent = False
        self.state = JobState.QUEUED
        self.accepted = asyncio.get_running_loop().create_future()
        self.result = asyncio.get_running_loop().create_future()
        self.model = None
        self.identified = asyncio.Event()
        self.stream_error = None
        # Machines still owing a terminal event; settings and downloads go to all.
        self.machines: set[str] = set()
        self.worker: str | None = None
        self.outcome: JsonValue = None
        self.failure: InferenceFailure | None = None
        self.image_progress = (
            ImageProgress(
                image_index=1,
                image_count=request.input.count,
                steps_total=request.input.steps,
            )
            if request.feature == "image"
            else None
        )
        self.progress: InferenceProgress | None = None
        self.events = asyncio.Queue(maxsize=64) if streaming else None
