"""Protocol-v5 FIFO worker with kill-and-join cancellation of its executor."""

import asyncio
import csv
import importlib
import json
import logging
import multiprocessing
import os
import resource
import socket
import subprocess
import tempfile
import time
from contextlib import suppress
from pathlib import Path

from experimental_worker.store import ModelStore

GIB = 1024**3
log = logging.getLogger(__name__)
BACKENDS = {
    "text": ("lfm3_llm", "HIKARI_ENABLE_LFM3_TEXT"),
    "vision": ("lfm3_vision", "HIKARI_ENABLE_LFM3_VISION"),
    "audio": ("lfm3_audio", None),
    "d1": ("lfm3_decision", "HIKARI_ENABLE_D1_DECISIONS"),
}


def enabled(name: str) -> bool:
    return os.getenv(name, "").lower() in ("1", "true", "yes")


def machine(cpu_only: bool = False) -> dict:
    """Match native worker boot/GPU identities and reserve its existing margins."""
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, amount, *_ = line.split()
        mem[name.rstrip(":")] = int(amount) * 1024
    total, free = mem["MemTotal"], mem["MemAvailable"]
    physical_total, physical_free = total, free
    ram_limit = None
    limit = Path("/sys/fs/cgroup/memory.max")
    current = Path("/sys/fs/cgroup/memory.current")
    if limit.exists() and limit.read_text().strip() != "max":
        maximum = int(limit.read_text())
        ram_limit = max(0, maximum - GIB)
        total = min(total, maximum)
        free = min(free, max(0, maximum - int(current.read_text())))
    devices = []
    if not cpu_only:
        rows = list(
            csv.reader(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=uuid,memory.total,memory.free",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).splitlines()
            )
        )
        index = int(os.getenv("HIKARI_EXPERIMENTAL_GPU", "0"))
        uuid, gpu_total, gpu_free = rows[index]
        devices = [
            {
                "index": 0,
                "uuid": uuid.strip(),
                "vram_total": int(gpu_total) * 1024**2,
                "vram_free": max(0, int(gpu_free) * 1024**2 - GIB // 2),
            }
        ]
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    return {
        "id": boot,
        "name": os.getenv("HIKARI_WORKER_MACHINE", "computer-" + boot[:8]),
        "ram_total": total,
        "ram_free": max(0, free - GIB),
        "ram_limit": ram_limit,
        "physical_ram_total": physical_total,
        "physical_ram_free": max(0, physical_free - GIB),
        # This trial uses one whole GPU; the hub must not pool several GPUs.
        "devices": devices,
    }


def execute_child(
    connection, backend_name, metadata, manifest, root, request, device, budget, scratch
):
    """The subprocess is the allocation boundary; exit releases its resources."""

    def emit(message):
        connection.send(message)

    try:
        tempfile.tempdir = scratch
        started = time.perf_counter()
        store = ModelStore(Path(root), metadata, manifest)
        if request["operation"] == "download":
            result = store.download(
                request["input"].get("license_acknowledged", False), emit
            )
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = device["uuid"] if device else ""
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            if device is not None:
                # Inference libraries must exist only in the disposable executor.
                import torch

                fraction = max(0, budget[1] - GIB) / device["vram_total"]
                if fraction <= 0:
                    raise ValueError(
                        "Insufficient CUDA allocation after runtime headroom"
                    )
                torch.cuda.set_per_process_memory_fraction(min(1.0, fraction), device=0)
                torch.cuda.reset_peak_memory_stats(0)
            backend = importlib.import_module("experimental_worker." + backend_name)
            result = backend.execute(
                request["input"], request["operation"], str(store.root), emit
            )
            if device is not None:
                torch.cuda.synchronize()
            # Decision answers are a bare list in the API contract. Report their
            # measurements separately without altering the typed answer payload.
            timing = result.setdefault("timing", {}) if isinstance(result, dict) else {}
            timing.update(
                executor_seconds=time.perf_counter() - started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
            )
            if device is not None:
                timing.update(
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0),
                )
            # Operator measurements contain counts/times only, never prompts,
            # generated text, screenshots or audio. CUDA allocator peaks exclude
            # driver/context memory; the server trial also samples nvidia-smi.
            print(
                "LFM3_METRIC "
                + json.dumps(
                    {
                        "request_id": request["id"],
                        "model": metadata["id"],
                        "timing": timing,
                    }
                ),
                flush=True,
            )
        emit({"state": "result", "result": result})
    except BaseException as error:
        emit({"state": "error", "error": str(error) or type(error).__name__})
    finally:
        connection.close()


class Runtime:
    def __init__(self, backend_name: str, info: dict | None = None):
        backend_name = backend_name.removeprefix("experimental_worker.")
        self.kind, flag = BACKENDS[backend_name]
        if flag is not None and not enabled(flag):
            raise ValueError(f"Experimental worker requires {flag}=true")
        self.backend_name = backend_name
        self.cpu_only = backend_name == "d1"
        self.backend = importlib.import_module("experimental_worker." + backend_name)
        self.manifest = json.loads(
            (Path(__file__).parent / f"catalog_{backend_name}.json").read_text()
        )
        self.root = Path(os.getenv("KADAN_MODEL_DIR", "/var/lib/kadan/models"))
        self.store = ModelStore(self.root, self.backend.MODEL, self.manifest)
        self.info = info if info is not None else machine(cpu_only=self.cpu_only)
        self.probe_machine = info is None
        self.budget = [0, 0]
        self.queue = asyncio.Queue(maxsize=16)
        self.cancelled: set[str] = set()
        self.active: str | None = None
        self.child = None
        self.write_lock = asyncio.Lock()
        self.granted = asyncio.Event()
        self.socket = None

    async def send(self, message):
        async with self.write_lock:
            await self.socket.send(json.dumps(message, allow_nan=False))

    async def models(self):
        await self.send({"state": "models", "models": self.store.status()})

    async def status(self, event="updated"):
        await self.send(
            {
                "state": "worker",
                "worker": self.kind,
                "event": event,
                "loaded": any(self.budget),
                "budget": self.budget,
                "model": None,
            }
        )

    async def handle(self, message):
        command, request_id = message["command"], message["id"]
        if command == "cancel":
            self.cancelled.add(request_id)
            self.granted.set()
            return
        if command == "submit":
            task = message["input"]
            model = task.get("model_id") or task.get("model")
            if (
                message["operation"] not in ("configure",)
                and model != self.backend.MODEL["id"]
            ):
                await self.send(
                    {
                        "id": request_id,
                        "state": "rejected",
                        "error": "Explicit experimental model ID is required.",
                        "status_code": 422,
                    }
                )
                return
            if message["operation"] in ("load", "unload"):
                await self.send(
                    {
                        "id": request_id,
                        "state": "rejected",
                        "error": "Experimental models load per request; manual residency is unavailable.",
                        "status_code": 422,
                    }
                )
                return
            try:
                self.queue.put_nowait(message)
            except asyncio.QueueFull:
                await self.send(
                    {"id": request_id, "state": "rejected", "error": "queue_full"}
                )
            else:
                await self.send({"id": request_id, "state": "queued"})
            return
        reply = {"id": request_id, "state": "control", "worker": self.kind, "ok": True}
        if command == "requirement":
            model = message.get("model")
            if model not in (None, self.backend.MODEL["id"]):
                reply.update(
                    state="rejected", error="Unsupported model", status_code=422
                )
            else:
                required = self.backend.REQUIREMENT
                reply.update(
                    state="requirement",
                    model=self.backend.MODEL["id"],
                    minimum=list(required),
                    preferred=list(required),
                    device_floor=required[1],
                )
        elif command in ("load", "offload"):
            if self.active is not None and self.child is not None:
                reply["ok"] = False
            else:
                budget = list(message["budget"]) if command == "load" else [0, 0]
                # The hub serializes only devices visible to this worker. A CPU
                # grant has one host component; executors use [host, device].
                if self.cpu_only and len(budget) == 1:
                    budget.append(0)
                if (
                    len(budget) != 2
                    or any(type(value) is not int or value < 0 for value in budget)
                    or (self.cpu_only and budget[1] != 0)
                ):
                    reply.update(
                        state="rejected", error="Invalid memory grant", status_code=422
                    )
                else:
                    self.budget = budget
                    self.granted.set()
                    await self.status(command)
        elif command == "status":
            reply.update(
                state="status",
                workers=[
                    {
                        "worker": self.kind,
                        "loaded": any(self.budget),
                        "budget": self.budget,
                        "used": self.budget if self.child is not None else [0, 0],
                        "model": None,
                    }
                ],
            )
        elif command == "is_busy":
            reply["busy"] = self.active is not None
        else:
            reply.update(
                state="rejected", error="Unknown worker command", status_code=422
            )
        await self.send(reply)

    async def run_job(self, request):
        request_id = request["id"]

        async def event(state, **fields):
            await self.send({"id": request_id, "state": state, **fields})

        if request_id in self.cancelled:
            await event("cancelled")
            return
        if request["operation"] == "configure":
            # Native formatting/context settings are broadcast to all workers. They
            # never select or alter this explicit experimental checkpoint.
            if (
                request["input"].get("feature") == "model_context"
                and request["input"].get("model_id") == self.backend.MODEL["id"]
            ):
                await event(
                    "failed",
                    error="The experimental trial uses a fixed bounded context.",
                    status_code=422,
                )
            else:
                await event("succeeded", result=self.store.status())
            return
        if request["operation"] != "download":
            if not self.store.complete():
                await event(
                    "failed",
                    error="Model is not downloaded; review its license and use the model download action first.",
                    status_code=409,
                )
                return
            while True:
                self.granted.clear()
                if all(
                    actual >= need
                    for actual, need in zip(self.budget, self.backend.REQUIREMENT)
                ):
                    break
                if request_id in self.cancelled:
                    await event("cancelled")
                    return
                await event("waiting_for_resources")
                await self.granted.wait()
        if request_id in self.cancelled:
            await event("cancelled")
            return
        with tempfile.TemporaryDirectory(prefix="lfm3-executor-") as scratch:
            context = multiprocessing.get_context("spawn")
            reader, writer = context.Pipe(duplex=False)
            process = context.Process(
                target=execute_child,
                args=(
                    writer,
                    self.backend_name,
                    self.backend.MODEL,
                    self.manifest,
                    str(self.root),
                    request,
                    None if self.cpu_only else self.info["devices"][0],
                    self.budget,
                    scratch,
                ),
            )
            self.child = process
            result, error = None, None
            started = False
            try:
                await event("preparing", model=self.backend.MODEL["id"])
                process.start()
                started = True
                executor_started = time.monotonic()
                writer.close()
                await event("running", model=self.backend.MODEL["id"])
                received_result = False
                while process.is_alive() or reader.poll():
                    if request_id in self.cancelled:
                        break
                    if request["operation"] != "download" and process.is_alive():
                        limit = getattr(self.backend, "MAX_EXECUTOR_SECONDS", None)
                        if (
                            limit is not None
                            and time.monotonic() - executor_started >= limit
                        ):
                            error = "Experimental executor exceeded its time limit."
                            break
                        try:
                            resident = next(
                                line
                                for line in Path(f"/proc/{process.pid}/status")
                                .read_text()
                                .splitlines()
                                if line.startswith("VmRSS:")
                            )
                            if int(resident.split()[1]) * 1024 > self.budget[0]:
                                error = "Experimental executor exceeded its granted RAM budget."
                                break
                        except (OSError, StopIteration):
                            pass
                    if reader.poll():
                        try:
                            message = reader.recv()
                        except EOFError:
                            break
                        state = message.pop("state")
                        if state == "result":
                            result, received_result = message["result"], True
                        elif state == "error":
                            error = message["error"]
                        else:
                            await event(state, **message)
                    else:
                        await asyncio.sleep(0.02)
                if not received_result and error is None:
                    error = "Experimental executor exited without a result."
            finally:
                # No terminal event or release may precede physical process exit.
                async def join_executor():
                    if started and process.is_alive():
                        process.terminate()
                    deadline = asyncio.get_running_loop().time() + 10
                    while (
                        started
                        and process.is_alive()
                        and asyncio.get_running_loop().time() < deadline
                    ):
                        await asyncio.sleep(0.02)
                    if started and process.is_alive():
                        process.kill()
                        while process.is_alive():
                            await asyncio.sleep(0.02)
                    if started:
                        process.join()

                cleanup = asyncio.create_task(join_executor())
                interrupted = False
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        # Disconnect or shutdown can cancel an already stopping job.
                        # Keep its grant until the shielded cleanup actually finishes.
                        interrupted = True
                cleanup.result()
                self.child = None
                reader.close()
                if request["operation"] != "download":
                    self.budget = [0, 0]
                if interrupted:
                    raise asyncio.CancelledError()
        if request_id in self.cancelled:
            await event("cancelled")
        elif error is not None:
            await event("failed", error=error, status_code=502)
        else:
            await event("succeeded", model=self.backend.MODEL["id"], result=result)
        if request["operation"] != "download":
            # Settle the head job first, so its old pending count cannot make
            # the scheduler regrant an already completed per-request engine.
            await self.status("released")
        await self.models()

    async def consume(self):
        while True:
            request = await self.queue.get()
            self.active = request["id"]
            try:
                await self.run_job(request)
            except Exception as error:
                await self.send(
                    {
                        "id": self.active,
                        "state": "failed",
                        "error": str(error),
                        "status_code": 502,
                    }
                )
                if self.child is None:
                    await self.status("released")
            finally:
                self.cancelled.discard(self.active)
                self.active = None
                self.queue.task_done()

    async def serve(self):
        """Reconnect with capped backoff; the API owns replacement/requeue decisions."""
        from websockets.exceptions import ConnectionClosed, InvalidHandshake

        delay = 1
        while True:
            try:
                await self.serve_session()
            except (OSError, TimeoutError, ConnectionClosed, InvalidHandshake) as error:
                log.warning("Experimental worker disconnected: %s", error)
            # serve_session has joined its executor before this reset. Old jobs
            # cannot replay after a reconnect; only the API may resubmit them.
            self.budget = [0, 0]
            self.queue = asyncio.Queue(maxsize=16)
            self.cancelled.clear()
            self.active = None
            self.granted.clear()
            self.socket = None
            await asyncio.sleep(delay)
            delay = min(30, delay * 2)

    async def serve_session(self):
        # The transport dependency is optional on the native/API paths.
        from websockets.asyncio.client import connect

        if self.probe_machine:
            self.info = machine(cpu_only=self.cpu_only)
        token = os.getenv("HIKARI_WORKER_TOKEN", "insecure-api-key")
        async with connect(
            os.getenv("HIKARI_API_URL", "ws://hikari-api:8000/v1/workers"),
            additional_headers={"Authorization": f"Bearer {token}"},
            max_size=128 * 1024**2,
            ping_interval=20,
            ping_timeout=40,
        ) as connection:
            self.socket = connection
            await self.send(
                {
                    "state": "hello",
                    "protocol": 5,
                    "worker": self.kind,
                    "instance": socket.gethostname(),
                    "machine": self.info,
                }
            )
            await self.models()
            await self.send({"state": "ready", "protocol": 5})
            consumer = asyncio.create_task(self.consume())
            try:
                async for message in connection:
                    await self.handle(json.loads(message))
            finally:
                if self.active:
                    self.cancelled.add(self.active)
                self.granted.set()
                consumer.cancel()
                with suppress(asyncio.CancelledError):
                    await consumer
