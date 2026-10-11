"""Repeatable formatting, build, and unit-test commands."""

from pathlib import Path
import sys

import typer

from bin.process import ROOT, run

app = typer.Typer(no_args_is_help=True)

NATIVE_UNIT_TARGETS = [
    "generation-queue-tests",
    "weight-backing-tests",
    "resource-tests",
    "quantization-tests",
    "checkpoint-tests",
    "checkpoint-floats-tests",
    "image-weight-loading-tests",
    "cuda-plan-tests",
    "fp8-plan-tests",
    "placement-tests",
    "sequence-state-tests",
    "h3-profile-tests",
    "worker-compute-tests",
    "device-workspace-tests",
    "device-weight-tests",
    "kadan_auto_placement_test",
    "resident-protocol-tests",
    "inference-owner-tests",
    "native-text-tests",
]


@app.command("format")
def format_python(write: bool = False) -> None:
    """Check Python formatting, or apply it with --write."""
    run([sys.executable, "-m", "ruff", "format", *([] if write else ["--check"]), "."])


@app.command()
def api() -> None:
    """Run isolated API unit tests."""
    run([sys.executable, "-m", "unittest", "discover", "-s", "api/tests", "-v"])


@app.command("frontend-lint")
def frontend_lint() -> None:
    """Lint frontend source."""
    run(["npm", "--prefix", "frontend", "run", "lint"])


@app.command("frontend-build")
def frontend_build() -> None:
    """Type-check and build the frontend."""
    run(["npm", "--prefix", "frontend", "run", "build"])


@app.command("frontend-unit")
def frontend_unit() -> None:
    """Run frontend unit tests."""
    run(["npm", "--prefix", "frontend", "run", "test:unit"])


@app.command()
def native(
    build_dir: Path = ROOT / "build/native-unit", sanitize: bool = False
) -> None:
    """Build and run only CPU unit targets; never run inference."""
    build_dir = build_dir.resolve()
    flags = "-fsanitize=address,undefined -fno-sanitize-recover=all" if sanitize else ""
    run(
        [
            "cmake",
            "-S",
            ROOT / "native",
            "-B",
            build_dir,
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DKADAN_ENABLE_CUDA=OFF",
            "-DKADAN_BUILD_TOOLS=OFF",
            "-DKADAN_ENABLE_DECISIONS=OFF",
            "-DKADAN_ENABLE_H3_GENERATION=OFF",
            "-DKADAN_ENABLE_TTS_WORKER=OFF",
            "-DKADAN_ENABLE_IMAGE_GENERATION=OFF",
            "-DKADAN_ENABLE_WHISPER_BLAS=OFF",
            "-DKADAN_ENABLE_IMAGE_BLAS=OFF",
            "-DBUILD_TESTING=ON",
            f"-DCMAKE_CXX_FLAGS={flags}",
        ]
    )
    run(
        [
            "cmake",
            "--build",
            build_dir,
            "--target",
            *NATIVE_UNIT_TARGETS,
            "--parallel",
            "2",
        ]
    )
    run(["ctest", "--test-dir", build_dir, "--output-on-failure"])


@app.command()
def install() -> None:
    """Validate API imports without starting the application or loading a model."""
    # Application dependencies belong only to this explicit image-build check.
    from api.services.native_worker import NativeWorker
    from api.services.inference import InferenceService

    assert NativeWorker and InferenceService
    typer.echo(
        "Single native worker transport imports successfully; no model execution."
    )
