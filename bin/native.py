"""Metadata-only native inspection and memory planning."""

import json
from pathlib import Path
from typing import Annotated

import typer

from bin.cache_replay import replay_trace
from bin.process import ROOT, run

app = typer.Typer(no_args_is_help=True)
DEFAULT_BUILD = ROOT / "build/inference-worker-tools"


@app.command("cache-replay")
def cache_replay(trace: Path) -> None:
    """Compare tile cache policies on a complete native trace, without inference."""
    try:
        report = replay_trace(trace)
    except (TypeError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="trace") from error
    typer.echo(json.dumps(report, indent=2))


@app.callback()
def configuration(ctx: typer.Context, build_dir: Path = DEFAULT_BUILD) -> None:
    """Select the build directory used by native commands."""
    ctx.obj = build_dir.resolve()


def execute(build_dir: Path, name: str, arguments: list[str | Path | int]) -> None:
    executable = build_dir / "tools" / name
    if not executable.is_file():
        raise FileNotFoundError(
            f"{executable} is missing. Run bin/kadan.py native --build-dir {build_dir} build first."
        )
    # Keep caller-relative input/output paths meaningful when invoked outside the repository.
    run([executable, *map(str, arguments)], cwd=Path.cwd())


@app.command()
def build(
    ctx: typer.Context,
    jobs: Annotated[int, typer.Option(min=1)] = 2,
) -> None:
    """Compile metadata inspectors without CUDA or inference executables."""
    run(
        [
            "cmake",
            "-S",
            ROOT / "inference_worker",
            "-B",
            ctx.obj,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=OFF",
            "-DKADAN_BUILD_TOOLS=ON",
            "-DKADAN_ENABLE_CUDA=OFF",
            "-DKADAN_ENABLE_DECISIONS=OFF",
            "-DKADAN_ENABLE_H3_GENERATION=OFF",
            "-DKADAN_ENABLE_TTS_WORKER=OFF",
            "-DKADAN_ENABLE_WHISPER_BLAS=OFF",
            "-DKADAN_ENABLE_IMAGE_GENERATION=OFF",
            "-DKADAN_ENABLE_IMAGE_BLAS=OFF",
        ]
    )
    run(
        [
            "cmake",
            "--build",
            ctx.obj,
            "--target",
            "kadan-tools",
            "--parallel",
            str(jobs),
        ]
    )


@app.command("checkpoint-inspect")
def checkpoint_inspect(
    ctx: typer.Context, root: Path, shard: str, metadata_bytes: int
) -> None:
    """Inspect a safetensors shard using a bounded metadata budget."""
    execute(ctx.obj, "kadan-checkpoint-inspect", [root, shard, metadata_bytes])


@app.command("model-inspect")
def model_inspect(ctx: typer.Context, root: Path, metadata_bytes: int) -> None:
    """Inspect model architecture and checkpoint metadata."""
    execute(ctx.obj, "kadan-model-inspect", [root, metadata_bytes])


@app.command("model-plan")
def model_plan(
    ctx: typer.Context, root: Path, capacity: int, metadata_bytes: int
) -> None:
    """Plan model memory for a token capacity without loading weights."""
    execute(ctx.obj, "kadan-model-plan", [root, capacity, metadata_bytes])


@app.command("model-placement")
def model_placement(
    ctx: typer.Context,
    root: Path,
    metadata_bytes: int,
    host_bytes: int,
    staging_bytes: int,
    device_bytes: int,
    headroom_bytes: int,
    expert_slots: int,
    split_layer: int,
) -> None:
    """Plan a two-device model placement with explicit memory budgets."""
    execute(
        ctx.obj,
        "kadan-model-inspect",
        [
            root,
            metadata_bytes,
            host_bytes,
            staging_bytes,
            device_bytes,
            headroom_bytes,
            expert_slots,
            split_layer,
        ],
    )
