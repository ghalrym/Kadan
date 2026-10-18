"""Worker protocol models, outstanding jobs, and the local machine's native process."""

import asyncio
import json
import logging
import os
import signal
import sys
from collections import deque
from contextlib import suppress
from enum import StrEnum
from pathlib import Path
from typing import Literal

from api.pydantic_models.inference import (
    ImageProgress,
    InferenceProgress,
)
from api.pydantic_models.inference_task import InferenceTask
from api.pydantic_models.model_store import ModelStoreStatus
from api.services.inference_errors import InferenceFailure
from pydantic import BaseModel, ConfigDict, JsonValue

log = logging.getLogger(__name__)

# Native messages travel as WebSocket text frames; see the worker's own bounds.
MAX_MESSAGE_BYTES = 256 * 1024 * 1024


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


class WorkerExit(BaseModel):
    pid: int
    returncode: int
    signal: int | None = None
    stderr: str

    def describe(self) -> str:
        cause = (
            f"signal {self.signal} ({signal.strsignal(self.signal)})"
            if self.signal
            else f"code {self.returncode}"
        )
        return f"Inference worker {self.pid} exited with {cause}: {self.stderr}"


class InferenceJob:
    """An outstanding response and bounded streaming channel, never an executor."""

    def __init__(self, request: InferenceTask, streaming: bool):
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


class LocalMachine:
    """Runs this machine's native worker process, which connects back to the API.

    The process exits when its connection drops, so it is restarted until close().
    """

    def __init__(self, root: Path | None = None):
        self.root = root or (
            Path(
                os.getenv(
                    "HIKARI_MODEL_DIR",
                    os.getenv("KADAN_MODEL_DIR", "~/.local/share/kadan/models"),
                )
            )
            .expanduser()
            .resolve()
            / "inference"
        )
        self.process = None
        self.diagnostics = deque(maxlen=64)
        self.last_exit: WorkerExit | None = None
        self.closing = False
        self.supervisor: asyncio.Task | None = None

    def executable(self) -> Path:
        executable = Path(
            os.getenv(
                "HIKARI_INFERENCE_WORKER",
                os.getenv(
                    "KADAN_INFERENCE_WORKER", "/opt/kadan/bin/kadan-inference-worker"
                ),
            )
        )
        if not executable.is_absolute() or not executable.is_file():
            raise InferenceFailure("The inference worker executable is unavailable.")
        return executable

    async def launch(self, url: str, token: str):
        self.root.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["HIKARI_WORKER_TOKEN"] = token
        self.process = await asyncio.create_subprocess_exec(
            str(self.executable()),
            str(self.root / "worker.lock"),
            url,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            env=environment,
        )

    async def start(self, url: str, token: str):
        if self.supervisor is not None:
            raise RuntimeError("Local machine already started")
        await self.launch(url, token)
        self.supervisor = asyncio.create_task(self._supervise(url, token))

    async def _supervise(self, url: str, token: str):
        delay = 1
        while True:
            await self._stderr()
            returncode = await self.process.wait()
            self.last_exit = WorkerExit(
                pid=self.process.pid,
                returncode=returncode,
                signal=-returncode if returncode < 0 else None,
                stderr="".join(self.diagnostics)[-8192:],
            )
            # A crashed worker may leave an owned codec process holding stderr open.
            with suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
            if self.closing:
                return
            log.error("%s; restarting in %ss", self.last_exit.describe(), delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)
            if self.closing:
                return
            try:
                await self.launch(url, token)
            except Exception:
                log.exception("Inference worker restart failed")
                continue

    async def _stderr(self):
        pending = bytearray()
        oversized = False
        while data := await self.process.stderr.read(4096):
            self.diagnostics.append(data.decode("utf-8", errors="replace"))
            parts = data.split(b"\n")
            for index, part in enumerate(parts):
                if not oversized:
                    if len(pending) + len(part) > 4095:
                        pending.clear()
                        oversized = True
                    else:
                        pending.extend(part)
                if index < len(parts) - 1:
                    if not oversized and pending.startswith(
                        (
                            b"KADAN_IMAGE_TIMING ",
                            b"KADAN_ATEN_IMAGE_TIMING ",
                            b"KADAN_ATEN_IMAGE_BATCH ",
                            b"KADAN_ATEN_IMAGE_CACHE ",
                            b"KADAN_TEXT_TIMING ",
                        )
                    ):
                        # Persist opted-in native timings without changing UI progress.
                        with suppress(OSError, ValueError):
                            sys.stderr.write(
                                pending.decode("utf-8", errors="replace") + "\n"
                            )
                            sys.stderr.flush()
                    pending.clear()
                    oversized = False

    async def close(self):
        self.closing = True
        # Stop supervising first: the API closes worker connections before its
        # own shutdown, and a restart now would retry against a stopped API.
        if self.supervisor is not None:
            self.supervisor.cancel()
            with suppress(Exception, asyncio.CancelledError):
                await self.supervisor
        if self.process is None:
            return
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 30)
            except TimeoutError:
                pass
        # Also reaps any owned codec process left in the worker's group.
        with suppress(ProcessLookupError):
            os.killpg(self.process.pid, signal.SIGKILL)
        await self.process.wait()
        self.process = None
