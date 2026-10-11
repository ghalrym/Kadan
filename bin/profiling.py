"""Build, capture, and summarize developer CUDA profiles."""

import bisect
import json
import os
from pathlib import Path
import signal
import subprocess

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
            ROOT / "native",
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
