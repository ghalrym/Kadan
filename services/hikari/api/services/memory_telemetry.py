"""Physical host/device observations, independent of inference admission."""

import subprocess

from pydantic import BaseModel, Field


class MemoryReading(BaseModel):
    label: str
    used_bytes: int = Field(ge=0)
    total_bytes: int = Field(gt=0)


def physical_memory() -> tuple[list[MemoryReading], list[str]]:
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
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=1,
        )
        devices = []
        for line in result.stdout.splitlines():
            device, used, total = (int(value.strip()) for value in line.split(","))
            if device < 0 or used > total:
                raise ValueError("Invalid physical GPU memory reading")
            devices.append(
                MemoryReading(
                    label=f"GPU {device} memory used",
                    used_bytes=used * 1024**2,
                    total_bytes=total * 1024**2,
                )
            )
        if not devices:
            raise ValueError("No GPU memory readings")
        readings.extend(devices)
    except (OSError, ValueError, subprocess.SubprocessError):
        errors.append("Physical GPU memory measurement unavailable")
    return readings, errors
