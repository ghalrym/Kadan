"""Repeatable formatting, build, and unit-test commands."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer

from bin.process import REPOSITORY, ROOT, run

app = typer.Typer(no_args_is_help=True)

INFERENCE_WORKER_UNIT_TARGETS = [
    "bf16-projection-tests",
    "media-session-tests",
    "media-residency-tests",
    "weight-backing-tests",
    "resource-tests",
    "quantization-tests",
    "checkpoint-tests",
    "checkpoint-floats-tests",
    "projection-validation-tests",
    "h3-resident-block-tests",
    "image-activation-tests",
    "image-input-tests",
    "image-reference-tests",
    "image-schedule-tests",
    "image-checkpoint-config-tests",
    "image-upload-tests",
    "image-edit-tests",
    "image-aten-plan-tests",
    "image-timing-tests",
    "image-decoder-input-tests",
    "image-convolution-tests",
    "mlx-quantization-tests",
    "gguf-quantization-tests",
    "gguf-checkpoint-tests",
    "gguf-text-tests",
    "h3-profile-tests",
    "worker-compute-tests",
    "device-workspace-tests",
    "device-weight-tests",
    "packed-read-team-tests",
    "device-weight-trace-tests",
    "inference-owner-tests",
    "preparation-tests",
    "s1-mini-metadata-tests",
    "text-tokenizer-tests",
    "inference-worker-text-tests",
]

CPP_SUFFIXES = {".cpp", ".hpp", ".h", ".cu", ".cuh"}

# CPU components only; the ATen image runtime requires the CUDA build.
CPP_LINT_OPTIONS = [
    "-DKADAN_ENABLE_CUDA=OFF",
    "-DKADAN_TRACE_WEIGHT_CACHE=ON",
    "-DKADAN_BUILD_TOOLS=ON",
    "-DKADAN_ENABLE_DECISIONS=ON",
    "-DKADAN_ENABLE_H3_GENERATION=ON",
    "-DKADAN_ENABLE_TTS_WORKER=ON",
    "-DKADAN_ENABLE_IMAGE_GENERATION=OFF",
    "-DBUILD_TESTING=ON",
]


def llvm_tool(name: str) -> str:
    """Prefer the pinned wheel installed beside this interpreter over the system tool."""
    pinned = Path(sys.executable).parent / name
    return str(pinned) if pinned.exists() else name


def cpp_sources() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", "inference_worker", "bin/cpp"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    return [
        ROOT / name
        for name in listed.split("\0")
        if Path(name).suffix in CPP_SUFFIXES and (ROOT / name).is_file()
    ]


@app.command("format")
def format_python(write: bool = False) -> None:
    """Check Python formatting, or apply it with --write."""
    run(
        [sys.executable, "-m", "ruff", "format", *([] if write else ["--check"]), "."],
        cwd=REPOSITORY,
    )


@app.command("cpp-format")
def format_cpp(write: bool = False) -> None:
    """Check C++ and CUDA formatting, or apply it with --write."""
    mode = ["-i"] if write else ["--dry-run", "--Werror"]
    run([llvm_tool("clang-format"), *mode, *cpp_sources()])


@app.command("cpp-lint")
def lint_cpp(
    build_dir: Path = ROOT / "build/inference-worker-lint", fix: bool = False
) -> None:
    """Run clang-tidy over every CPU translation unit, or apply its fixes with --fix."""
    build_dir = build_dir.resolve()
    run(
        [
            "cmake",
            "-S",
            ROOT / "inference_worker",
            "-B",
            build_dir,
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            *CPP_LINT_OPTIONS,
        ]
    )
    sources = set(cpp_sources())
    commands = json.loads((build_dir / "compile_commands.json").read_text())
    units = sorted({Path(c["file"]) for c in commands} & sources)
    tidy = [
        llvm_tool("clang-tidy"),
        "-p",
        str(build_dir),
        "--quiet",
        # CMake omits -std when the compiler already defaults to C++20, which clang does not.
        "--extra-arg=-std=gnu++20",
        *(["--fix"] if fix else []),
    ]

    def check_unit(unit: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*tidy, str(unit)], cwd=ROOT, capture_output=True, text=True, check=False
        )

    # Fixes to a shared header must not race between translation units.
    workers = 1 if fix else os.cpu_count() or 1
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(check_unit, units))
    failed = [result for result in results if result.returncode]
    for result in results:
        if result.stdout.strip():
            typer.echo(result.stdout, nl=False)
        if result.returncode and result.stderr.strip():
            typer.echo(result.stderr, err=True, nl=False)
    typer.echo(f"clang-tidy checked {len(units)} translation units.")
    if failed:
        raise typer.Exit(1)


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
def inference_worker(
    build_dir: Path = ROOT / "build/inference-worker-unit", sanitize: bool = False
) -> None:
    """Build and run only CPU unit targets; never run inference."""
    build_dir = build_dir.resolve()
    flags = "-fsanitize=address,undefined -fno-sanitize-recover=all" if sanitize else ""
    run(
        [
            "cmake",
            "-S",
            ROOT / "inference_worker",
            "-B",
            build_dir,
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DKADAN_ENABLE_CUDA=OFF",
            "-DKADAN_TRACE_WEIGHT_CACHE=ON",
            "-DKADAN_BUILD_TOOLS=OFF",
            "-DKADAN_ENABLE_DECISIONS=OFF",
            "-DKADAN_ENABLE_H3_GENERATION=OFF",
            "-DKADAN_ENABLE_TTS_WORKER=OFF",
            "-DKADAN_ENABLE_IMAGE_GENERATION=OFF",
            "-DKADAN_ENABLE_WHISPER_BLAS=OFF",
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
            *INFERENCE_WORKER_UNIT_TARGETS,
            "--parallel",
            "2",
        ]
    )
    run(["ctest", "--test-dir", build_dir, "--output-on-failure"])
    run([sys.executable, "-m", "unittest", "discover", "-s", "bin/tests", "-v"])


@app.command()
def install() -> None:
    """Validate API imports without starting the application or loading a model."""
    # Application dependencies belong only to this explicit image-build check.
    from api.services.inference import InferenceService
    from api.services.inference_worker import InferenceWorker

    assert InferenceWorker and InferenceService
    typer.echo("Inference worker transport imports successfully; no model execution.")
