"""Hot-reloading processes: the API with reload, and each worker program rebuilt on source edits."""

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

# Each worker image keeps its own build tree; see
# inference_worker/workers/<name>/Dockerfile.
WORKER_SOURCE = Path(os.getenv("HIKARI_WORKER_SOURCE", "/src/inference_worker"))
WORKER_BUILD = Path(os.getenv("HIKARI_WORKER_BUILD", "/build"))
WORKER_PREFIX = Path(os.getenv("HIKARI_WORKER_PREFIX", "/opt/kadan"))
NATIVE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".cmake", ".sh"}
WORKERS = ("decision", "speech", "image", "video")


def environment() -> dict[str, str]:
    """Expose the toolchain's CUDA libraries to the worker process."""
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
    """Worker sources need a rebuild; unit tests are not part of a worker binary."""
    if path.parts[0] == "tests":
        return False
    return path.name == "CMakeLists.txt" or path.suffix in NATIVE_SUFFIXES


def data(path: Path) -> bool:
    """A worker reads its catalog and assets at startup, so they need only a restart."""
    return path.name == "catalog.json" or path.parts[0] == "assets"


def target(name: str) -> str:
    return f"kadan-worker-{name}"


def installed(name: str) -> Path:
    return WORKER_PREFIX / "libexec" / target(name)


def fingerprint(name: str) -> int | None:
    """The installed binary's modification time; unchanged builds keep it."""
    try:
        return installed(name).stat().st_mtime_ns
    except FileNotFoundError:
        return None


def build(env: dict[str, str], name: str) -> bool:
    """Incrementally rebuild and install one worker, keeping the previous one on failure."""
    commands = [
        [
            "cmake",
            "--build",
            WORKER_BUILD,
            "--target",
            target(name),
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
            target(name),
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
                f"{target(name)} build failed; keeping the previous worker.", err=True
            )
            return False
    return True


def start_worker(env: dict[str, str], name: str) -> subprocess.Popen:
    return subprocess.Popen([str(WORKER_PREFIX / "bin" / target(name))], env=env)


def start_api(env: dict[str, str]) -> subprocess.Popen:
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
            # Worker results, such as finished videos, are large WebSocket messages.
            "--ws-max-size",
            str(256 * 1024 * 1024),
        ],
        cwd=ROOT,
        env=env,
    )


def stop(process: subprocess.Popen) -> None:
    """Stop a child gracefully; a worker releases its engine before exiting."""
    if process.poll() is None:
        process.terminate()
        process.wait()


@app.command()
def serve() -> None:
    """Serve the API with reload on source edits. Workers run in their own containers."""
    # Container stops signal only this process; unwind so the API shuts down cleanly.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    server = start_api(os.environ.copy())
    try:
        server.wait()
    finally:
        stop(server)


@app.command()
def worker(name: str) -> None:
    """Run one worker program, rebuilding and restarting only it when its sources change."""
    if name not in WORKERS:
        raise typer.BadParameter(f"Choose one of: {', '.join(WORKERS)}")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    env = environment()
    known = snapshot()
    # A failed first build still runs the previous worker; the next fixing edit retries.
    build(env, name)
    process = start_worker(env, name)
    try:
        while True:
            time.sleep(1)
            # Check edits even after an early exit so a source fix can recover it.
            restart = process.poll() is not None
            current = snapshot()
            if current != known:
                current = settled(current)
                changed = {
                    path
                    for path in known.keys() | current.keys()
                    if known.get(path) != current.get(path)
                }
                known = current
                if any(native(path) for path in changed):
                    before = fingerprint(name)
                    rebuilt = build(env, name) and fingerprint(name) != before
                    restart = restart or rebuilt
                elif any(data(path) for path in changed):
                    restart = True
            if not restart:
                continue
            typer.echo(f"Restarting {target(name)}.", err=True)
            stop(process)
            process = start_worker(env, name)
    finally:
        stop(process)
