"""Build, capture, and summarize developer CUDA profiles."""

import bisect
import json
import os
import signal
import statistics
import subprocess
from pathlib import Path

import typer

from bin.process import ROOT, run

app = typer.Typer(no_args_is_help=True)
DEFAULT_BUILD = ROOT / "build/profiling"


@app.command()
def build(build_dir: Path = DEFAULT_BUILD) -> None:
    """Compile the CUDA interposer without executing a GPU workload."""
    build_dir = build_dir.resolve()
    run(
        [
            "cmake",
            "-S",
            ROOT / "inference_worker",
            "-B",
            build_dir,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=OFF",
            "-DKADAN_BUILD_TOOLS=ON",
            "-DKADAN_ENABLE_CUDA=ON",
        ]
    )
    run(
        [
            "cmake",
            "--build",
            build_dir,
            "--target",
            "kadan-cuda-phase-profile",
            "--parallel",
            "2",
        ]
    )


@app.command()
def phase(pid: int) -> None:
    """Mark the next profile phase while the profiled worker is idle."""
    if pid <= 0:
        raise typer.BadParameter("PID must identify one process (a positive integer).")
    os.kill(pid, signal.SIGUSR1)


@app.command()
def symbols(executable: Path, output: Path) -> None:
    """Write the matching executable's demangled symbol table."""
    with output.open("w") as stream:
        subprocess.run(["nm", "-n", "-C", str(executable)], stdout=stream, check=True)


def summarize_profile(symbols: Path, report: Path) -> dict:
    """Resolve captured callsite offsets and sort kernels by measured GPU time."""
    entries = []
    for line in symbols.read_text().splitlines():
        parts = line.strip().split(" ", 2)
        if len(parts) == 3 and parts[1] in ("t", "T"):
            entries.append((int(parts[0], 16), parts[2]))
    entries.sort()
    if not entries:
        raise ValueError("The symbol table contains no text symbols.")
    addresses = [address for address, _ in entries]
    kernels, totals = [], []
    for line in report.read_text().splitlines():
        parts = line.strip().split("\t")
        if not line.strip() or parts[0] == "kernel_offset":
            continue
        if len(parts) != 4:
            totals.append(parts)
            continue
        address = int(parts[0], 16)
        index = bisect.bisect_right(addresses, address) - 1
        if index < 0:
            raise ValueError(
                f"No symbol precedes callsite {parts[0]}; use the matching executable."
            )
        kernels.append(
            dict(
                name=entries[index][1],
                calls=int(parts[1]),
                gpu_ms=float(parts[2]),
                launch_host_ms=float(parts[3]),
            )
        )
    kernels.sort(key=lambda row: -row["gpu_ms"])
    return dict(kernels=kernels, totals=totals)


@app.command()
def summarize(symbols: Path, report: Path, output: Path | None = None) -> None:
    """Convert a symbol table and phase TSV into a JSON report."""
    result = json.dumps(summarize_profile(symbols, report), indent=2) + "\n"
    if output is None:
        typer.echo(result, nl=False)
    else:
        output.write_text(result)


def projection_samples(report: Path) -> dict:
    """Extract aggregate phase records without copying unrelated log/payload fields."""
    samples = []
    for line in report.read_text().splitlines():
        marker = "KADAN_TEXT_TIMING "
        if marker not in line:
            continue
        record = json.loads(line.split(marker, 1)[1])
        detail = record.get("projection_detail")
        if detail is None:
            continue
        if detail.get("version") != 1:
            raise ValueError("Unsupported projection timing version.")
        samples.append(
            {
                key: record[key]
                for key in (
                    "phase",
                    "tokens",
                    "seconds",
                    "projection_seconds",
                    "attention_seconds",
                    "recurrent_seconds",
                    "device_seconds",
                    "projections",
                    "projection_batches",
                    "streamed_bytes",
                    "host_expert_bytes",
                    "packed_bytes",
                    "device_bytes",
                    "device_slices",
                    "ram_bytes",
                    "ram_slices",
                    "prepare_seconds",
                    "host_wait_seconds",
                    "projection_detail",
                )
                if key in record
            }
        )
    if not samples:
        raise ValueError("No opted-in projection samples found.")
    phases = {}
    for phase in sorted({sample["phase"] for sample in samples}):
        selected = [sample for sample in samples if sample["phase"] == phase]
        series = {
            "seconds": [sample["seconds"] for sample in selected],
            "projection_seconds": [sample["projection_seconds"] for sample in selected],
            "batch_wall_seconds": [
                sample["projection_detail"]["batch_wall_seconds"] for sample in selected
            ],
            "source_wall_seconds": [
                sum(
                    group["source_wall_seconds"]
                    for group in sample["projection_detail"]["groups"]
                )
                for sample in selected
            ],
        }
        phases[phase] = {
            "samples": len(selected),
            "spread": {
                name: {
                    "min": min(values),
                    "median": statistics.median(values),
                    "max": max(values),
                }
                for name, values in series.items()
            },
        }
    return {
        "samples": samples,
        "phases": phases,
        "timing": "completed-call wall seconds, not GPU time",
    }


@app.command("projections")
def projections(report: Path, output: Path) -> None:
    """Extract projection phase samples from a saved worker log; runs no inference."""
    output.write_text(json.dumps(projection_samples(report), indent=2) + "\n")
