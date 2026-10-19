"""Physical host/device observations, independent of inference admission."""

import asyncio
from collections.abc import Mapping

from pydantic import BaseModel, Field, model_validator

from api.services.worker_hub import Worker


class MemoryReading(BaseModel):
    label: str
    used_bytes: int = Field(ge=0)
    total_bytes: int = Field(gt=0)


class GpuMemory(BaseModel):
    uuid: str = Field(min_length=1)
    vram_used: int = Field(ge=0, strict=True)
    vram_total: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def valid_usage(self):
        if self.vram_used > self.vram_total:
            raise ValueError("Invalid physical GPU memory reading")
        return self


async def physical_memory(
    workers: Mapping[str, Worker],
) -> tuple[list[MemoryReading], list[str]]:
    readings, errors = [], []
    try:
        with open("/proc/meminfo") as stream:
            values = {
                parts[0].rstrip(":"): int(parts[1]) * 1024
                for line in stream
                if (parts := line.split())
                and parts[0] in ("MemTotal:", "MemAvailable:")
            }
        total, available = values["MemTotal"], values["MemAvailable"]
        readings.append(
            MemoryReading(
                label="System RAM used",
                used_bytes=max(0, total - available),
                total_bytes=total,
            )
        )
    except (OSError, KeyError, ValueError):
        errors.append("Host RAM measurement unavailable")

    # Status uses the workers' existing NVML access, never a CUDA context or
    # the scheduler's free-memory margins. Replicas observe the same devices.
    sources = [worker for worker in workers.values() if worker.devices]
    ready = [worker for worker in sources if worker.ready.is_set()]
    replies = await asyncio.gather(
        *(asyncio.wait_for(worker.control("status"), timeout=2) for worker in ready),
        return_exceptions=True,
    )
    expected = {
        (worker.computer.id, uuid)
        for worker in sources
        for uuid, index in worker.computer.gpus.items()
        if index in worker.devices
    }
    observed = {}
    computers = {worker.computer.id for worker in sources}
    for worker, reply in zip(ready, replies):
        if isinstance(reply, BaseException) or workers.get(worker.id) is not worker:
            continue
        try:
            machine = reply["machine"]
            if machine["id"] != worker.computer.id:
                raise ValueError("Unexpected computer identity")
            devices = [
                GpuMemory.model_validate(device) for device in machine["devices"]
            ]
            for device in devices:
                key = (worker.computer.id, device.uuid)
                index = worker.computer.gpus.get(device.uuid)
                if index not in worker.devices or key in observed:
                    continue
                prefix = (
                    f"{worker.computer.name} ({worker.computer.id[:8]}) "
                    if len(computers) > 1
                    else ""
                )
                observed[key] = MemoryReading(
                    label=f"{prefix}GPU {index - 1} memory used",
                    used_bytes=device.vram_used,
                    total_bytes=device.vram_total,
                )
        except (KeyError, TypeError, ValueError):
            continue
    readings.extend(observed[key] for key in sorted(observed))
    if not expected or expected - observed.keys():
        errors.append("Physical GPU memory measurement unavailable")
    return readings, errors
