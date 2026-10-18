"""Development server that keeps the API and its inference worker current with source edits."""

import os
import signal
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

import typer

from bin.process import ROOT

app = typer.Typer(no_args_is_help=True)

# The development image keeps the worker's release build tree; see api/Dockerfile.
WORKER_SOURCE = Path(os.getenv("HIKARI_WORKER_SOURCE", "/src/inference_worker"))
WORKER_BUILD = Path(os.getenv("HIKARI_WORKER_BUILD", "/build"))
WORKER_PREFIX = Path(os.getenv("HIKARI_WORKER_PREFIX", "/opt/kadan"))
NATIVE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".cmake", ".sh"}


def environment() -> dict[str, str]:
    """Expose the toolchain's CUDA libraries, which release images copy beside the worker."""
    env = os.environ.copy()
    nvidia = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    libraries = [str(path) for path in sorted(nvidia.glob("*/lib"))]
    if env.get("LD_LIBRARY_PATH"):
        libraries.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(libraries)
    # Read the catalog and assets from the source mount instead of the installed copy.
    env["KADAN_WORKER_DATA"] = str(WORKER_SOURCE)
    return env


def snapshot() -> dict[Path, int]:
    times = {}
    for path in WORKER_SOURCE.rglob("*"):
        # Editors replace files while saving; a vanished file is picked up next scan.
        try:
            if path.is_file():
                times[path.relative_to(WORKER_SOURCE)] = path.stat().st_mtime_ns
        except FileNotFoundError:
            pass
    return times


def settled(previous: dict[Path, int]) -> dict[Path, int]:
    """Wait until a burst of saves stops so one edit triggers one rebuild."""
    while True:
        time.sleep(0.5)
        current = snapshot()
        if current == previous:
            return current
        previous = current


def native(path: Path) -> bool:
    """Worker sources need a rebuild; unit tests are not part of the worker binary."""
    if path.parts[0] == "tests":
        return False
    return path.name == "CMakeLists.txt" or path.suffix in NATIVE_SUFFIXES


def data(path: Path) -> bool:
    """The worker reads its catalog and assets at startup, so they need only a restart."""
    return path.name == "catalog.json" or path.parts[0] == "assets"


def build(env: dict[str, str]) -> bool:
    """Incrementally rebuild and install the worker, keeping the previous one on failure."""
    commands = [
        [
            "cmake",
            "--build",
            WORKER_BUILD,
            "--target",
            "kadan-inference-worker",
            "--parallel",
            str(os.cpu_count() or 2),
        ],
        [
            "cmake",
            "--install",
            WORKER_BUILD,
            "--prefix",
            WORKER_PREFIX,
            "--component",
            "inference-runtime",
        ],
    ]
    for command in commands:
        if (
            subprocess.run(
                [str(part) for part in command], env=env, check=False
            ).returncode
            != 0
        ):
            typer.echo(
                "Inference worker build failed; keeping the previous worker.", err=True
            )
            return False
    return True


def start(env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.server:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--reload",
            "--reload-dir",
            str(ROOT / "api"),
        ],
        cwd=ROOT,
        env=env,
    )


def stop(server: subprocess.Popen) -> None:
    """Shut the API down gracefully; its shutdown closes the owned inference worker."""
    if server.poll() is None:
        server.terminate()
        server.wait()


@app.command()
def serve() -> None:
    """Serve the API with reload, rebuilding the worker when its sources change."""
    # Container stops signal only this process; unwind so the API shuts down cleanly.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    env = environment()
    known = snapshot()
    # A failed first build still serves the API; the next fixing edit retries.
    build(env)
    server = start(env)
    try:
        while True:
            time.sleep(1)
            current = snapshot()
            if current == known:
                continue
            current = settled(current)
            changed = {
                path
                for path in known.keys() | current.keys()
                if known.get(path) != current.get(path)
            }
            known = current
            if any(native(path) for path in changed):
                typer.echo("Inference worker source changed; rebuilding.", err=True)
                if not build(env):
                    continue
            elif not any(data(path) for path in changed):
                continue
            typer.echo(
                "Restarting the API with the updated inference worker.", err=True
            )
            stop(server)
            server = start(env)
    finally:
        stop(server)
