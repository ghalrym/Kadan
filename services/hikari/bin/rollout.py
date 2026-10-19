"""Pinned, serialized Compose image replacement; never build on the server."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import typer
from pydantic import BaseModel, ConfigDict, model_validator

logger = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=True)
WORKERS = tuple(
    f"hikari-worker-{kind}"
    for kind in ("llm", "decision", "tts", "speech", "formatting", "image", "video")
)
SERVICES = (
    "hikari-api",
    "hikari-frontend",
    *WORKERS,
    "tsuchi-api",
    "tsuchi-frontend",
    "kadan-api",
    "kadan-runner",
    "kadan-frontend",
)
OWNERS = ("kadan-api", "kadan-runner", "hikari-api", "tsuchi-api")
PIN = re.compile(r"[a-z0-9][a-z0-9./_-]*@sha256:[a-f0-9]{64}\Z")
SHA = re.compile(r"[a-f0-9]{40}\Z")


class Release(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: str
    compose_sha256: str
    compatibility: str
    images: dict[str, str]

    @model_validator(mode="after")
    def validate_release(self):
        if not SHA.fullmatch(self.source):
            raise ValueError("Release source must be a full commit SHA")
        if not re.fullmatch(r"[a-f0-9]{64}", self.compose_sha256):
            raise ValueError("Release must pin the Compose topology hash")
        if not self.compatibility or set(self.images) != set(SERVICES):
            raise ValueError(
                "Release must name its compatibility epoch and every service"
            )
        if any(not PIN.fullmatch(pin) for pin in self.images.values()):
            raise ValueError("Every image must use an immutable sha256 digest")
        if self.images["kadan-api"] != self.images["kadan-runner"]:
            raise ValueError("Kadan API and runner share one image")
        if any(
            not pin.startswith("ghcr.io/ghalrym/kadan/") for pin in self.images.values()
        ):
            raise ValueError("Images must belong to the Kadan release registry")
        return self

    @classmethod
    def read(cls, path: Path) -> Release:
        return cls(**json.loads(path.read_text()))


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def locked(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


class Compose:
    def __init__(self, checkout: Path, state: Path, timeout: int = 300):
        self.checkout = checkout.resolve()
        self.state = state.resolve()
        self.timeout = timeout
        self.installation = json.loads((state / "installation.json").read_text())
        self.project = self.installation["project"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.project):
            raise ValueError("Invalid enrolled Compose project")

    def command(self, *args: str) -> str:
        result = subprocess.run(
            [
                "docker",
                "--host",
                "unix:///var/run/docker.sock",
                "compose",
                "--project-name",
                self.project,
                "--env-file",
                str(self.checkout / ".env"),
                "--project-directory",
                str(self.checkout),
                "-f",
                str(self.checkout / "compose.yaml"),
                "-f",
                str(self.state / "images.json"),
                *args,
            ],
            cwd=self.checkout,
            check=True,
            text=True,
            capture_output=True,
            timeout=self.timeout,
            env={
                key: value
                for key, value in os.environ.items()
                if key in {"PATH", "HOME", "DOCKER_CONFIG", "TMPDIR"}
            },
        )
        return result.stdout

    def overlay(self, release: Release) -> None:
        services = {name: {"image": image} for name, image in release.images.items()}
        # The migration image is pinned too, but automatic rollout never runs it.
        services["hikari-migrate"] = {"image": release.images["hikari-api"]}
        for name in OWNERS:
            services[name]["volumes"] = [
                {
                    "type": "bind",
                    "source": str(self.state / "gate"),
                    "target": "/run/kadan/rollout",
                    "read_only": True,
                }
            ]
            services[name]["environment"] = {"KADAN_ROLLOUT_SERVICE": name}
        atomic_json(self.state / "images.json", {"services": services})

    def preflight(self, current: Release, target: Release) -> None:
        if any(
            current.images[name] != target.images[name]
            for name in ("hikari-api", *WORKERS)
        ):
            raise ValueError(
                "Native queue snapshot protocol required for Hikari image changes"
            )
        if current.source == target.source and current != target:
            raise ValueError(
                "A commit cannot be republished with different release contents"
            )
        if current.compatibility != target.compatibility:
            raise ValueError("Compatibility epoch changed: manual migration required")
        self.configuration(current)
        if target.compose_sha256 != current.compose_sha256:
            raise ValueError(
                "Release Compose topology changed: manual rollout required"
            )
        self.assert_running(current)
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", current.source, target.source],
            cwd=self.checkout,
            check=True,
            capture_output=True,
        )
        changes = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                current.source,
                target.source,
                "--",
                "services/hikari/api/migrations",
                "services/tsuchi/tsuchi/repository",
                "services/tsuchi/tsuchi/repository.py",
            ],
            cwd=self.checkout,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if changes.strip():
            raise ValueError("Storage migration changes require a manual rollout")

    def configuration(self, release: Release) -> None:
        topology = hashlib.sha256(
            (self.checkout / "compose.yaml").read_bytes()
        ).hexdigest()
        environment = hashlib.sha256((self.checkout / ".env").read_bytes()).hexdigest()
        if topology != release.compose_sha256:
            raise ValueError("Compose topology changed: manual rollout required")
        if environment != self.installation["env_sha256"]:
            raise ValueError("Enrolled environment changed: manual rollout required")

    def assert_running(self, release: Release) -> None:
        ids = self.command("ps", "--all", "--quiet", *SERVICES).split()
        if not ids:
            raise ValueError("Installation is not enrolled and running")
        values = json.loads(
            subprocess.run(
                ["docker", "--host", "unix:///var/run/docker.sock", "inspect", *ids],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        observed = {}
        for value in values:
            name = value["Config"]["Labels"]["com.docker.compose.service"]
            if name in observed or not value["State"]["Running"]:
                raise ValueError("Unsupported replica topology or stopped container")
            observed[name] = value["Config"]["Image"]
        if observed != release.images:
            raise ValueError(
                "Live images differ from last-known-good pins; manual enrollment required"
            )

    def replace(self, release: Release, services: tuple[str, ...]) -> None:
        self.overlay(release)
        self.command("pull", *services)
        self.command(
            "up",
            "-d",
            "--no-deps",
            "--no-build",
            "--pull",
            "never",
            "--wait",
            "--wait-timeout",
            str(self.timeout),
            *services,
        )

    def status(self, service: str) -> dict:
        return json.loads(
            self.command(
                "exec", "-T", service, "python", "-m", "kadan_tracing.rollout", "status"
            )
        )

    def drain(self, allow_stopped: tuple[str, ...] = ()) -> None:
        closed = json.loads((self.state / "gate" / "fence.json").read_text())["closed"]
        if not isinstance(closed, list) or not set(closed) <= set(OWNERS):
            raise ValueError("Invalid admission fence")
        for stage in (("kadan-api", "kadan-runner"), ("hikari-api",), ("tsuchi-api",)):
            closed.extend(service for service in stage if service not in closed)
            fence(self.state, list(closed))
            deadline = time.monotonic() + self.timeout
            while True:
                statuses = []
                for service in stage:
                    try:
                        statuses.append(self.status(service))
                    except (subprocess.SubprocessError, OSError, ValueError):
                        if service not in allow_stopped or not self.stopped(service):
                            raise
                        statuses.append({"fenced": True, "active": 0})
                if all(value["fenced"] and value["active"] == 0 for value in statuses):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Admission did not drain; containers must stay running"
                    )
                time.sleep(0.2)

    def stop_failed_replacements(self, journal: dict) -> None:
        closed = json.loads((self.state / "gate" / "fence.json").read_text())["closed"]
        if set(closed) != set(OWNERS):
            raise RuntimeError(
                "Replacement recovery requires the original complete fence"
            )
        target = Release(**journal["target"])
        for service in journal["affected"]:
            if service not in OWNERS:
                continue
            try:
                self.status(service)
                continue
            except (subprocess.SubprocessError, OSError, ValueError):
                pass
            ids = self.command("ps", "--all", "--quiet", service).split()
            if not ids:
                continue
            values = json.loads(
                subprocess.run(
                    [
                        "docker",
                        "--host",
                        "unix:///var/run/docker.sock",
                        "inspect",
                        *ids,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            )
            for value in values:
                config = value["Config"]
                mounted = any(
                    mount["Source"] == str(self.state / "gate")
                    and mount["Destination"] == "/run/kadan/rollout"
                    and not mount["RW"]
                    for mount in value["Mounts"]
                )
                if (
                    config["Image"] != target.images[service]
                    or not mounted
                    or f"KADAN_ROLLOUT_SERVICE={service}" not in config["Env"]
                ):
                    raise RuntimeError(
                        "Cannot prove failed replacement started under the fence"
                    )
            # Journal + digest + immutable read-only gate prove this is a newly
            # started target owner which has never admitted work since old drain.
            self.command("stop", "--timeout", "10", service)

    def stopped(self, service: str) -> bool:
        ids = self.command("ps", "--all", "--quiet", service).split()
        if not ids:
            return True
        values = json.loads(
            subprocess.run(
                ["docker", "--host", "unix:///var/run/docker.sock", "inspect", *ids],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        # A restarting process can admit work again, so it is never considered idle.
        return all(
            value["State"]["Status"] in {"exited", "dead", "created"}
            and not value["State"]["Running"]
            for value in values
        )

    def verify(self, services: tuple[str, ...]) -> None:
        deadline = time.monotonic() + self.timeout
        while True:
            output = self.command("ps", "--format", "json", *services).strip()
            # Compose versions emit either an array or one JSON object per line.
            containers = (
                json.loads(output)
                if output.startswith("[")
                else [json.loads(line) for line in output.splitlines()]
            )
            ready = {value["Service"] for value in containers} == set(services)
            for container in containers:
                ready &= container["State"] == "running"
                if container.get("Health"):
                    ready &= container["Health"] == "healthy"
            statuses = {service: self.status(service) for service in OWNERS}
            ready &= all(
                value["fenced"] and value["ready"] and value["active"] == 0
                for value in statuses.values()
            )
            if ready:
                try:
                    for owner in OWNERS:
                        self.command(
                            "exec",
                            "-T",
                            owner,
                            "python",
                            "-m",
                            "kadan_tracing.rollout",
                            "smoke",
                        )
                    for frontend in (
                        "kadan-frontend",
                        "tsuchi-frontend",
                        "hikari-frontend",
                    ):
                        for path in (
                            "/",
                            "/health"
                            if frontend == "hikari-frontend"
                            else "/api/health",
                        ):
                            self.command(
                                "exec",
                                "-T",
                                frontend,
                                "wget",
                                "-q",
                                "-O",
                                "/dev/null",
                                "http://127.0.0.1:8080" + path,
                            )
                    return
                except subprocess.CalledProcessError:
                    pass
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Replacement health/readiness did not become healthy"
                )
            time.sleep(0.2)


def fence(state: Path, closed: list[str]) -> None:
    gate = state / "gate"
    gate.mkdir(exist_ok=True)
    atomic_json(gate / "fence.json", {"closed": closed})


def recover(state: Path, backend: Compose, journal: dict) -> None:
    if journal["phase"] not in {"draining", "replacing"}:
        raise ValueError("Unknown recovery phase; keep admission fenced")
    previous = Release(**journal["previous"])
    logger.warning("Recovering release %s phase %s", previous.source, journal["phase"])
    if journal["phase"] == "replacing":
        backend.stop_failed_replacements(journal)
        backend.drain(tuple(journal["affected"]))
        backend.replace(previous, tuple(journal["affected"]))
        backend.verify(tuple(journal["affected"]))
    atomic_json(state / "state.json", {"good": previous.model_dump(), "pending": None})
    fence(state, [])


def rollout(target: Release, state: Path, backend: Compose) -> None:
    with locked(state):
        record = json.loads((state / "state.json").read_text())
        backend.configuration(Release(**record["good"]))
        if record["pending"]:
            recover(state, backend, record["pending"])
            record = json.loads((state / "state.json").read_text())
        current = Release(**record["good"])
        backend.preflight(current, target)
        gate = state / "gate" / "fence.json"
        if gate.exists() and json.loads(gate.read_text())["closed"]:
            backend.drain()
            backend.verify(SERVICES)
            fence(state, [])
        if current == target:
            return
        affected = tuple(
            name for name in SERVICES if current.images[name] != target.images[name]
        )
        if not affected:
            atomic_json(
                state / "state.json", {"good": target.model_dump(), "pending": None}
            )
            return
        journal = {
            "previous": current.model_dump(),
            "target": target.model_dump(),
            "affected": affected,
            "phase": "draining",
        }
        atomic_json(
            state / "state.json", {"good": current.model_dump(), "pending": journal}
        )
        try:
            backend.drain()
            journal["phase"] = "replacing"
            atomic_json(
                state / "state.json", {"good": current.model_dump(), "pending": journal}
            )
            logger.info(
                "Replacing release %s services %s", target.source, ",".join(affected)
            )
            backend.replace(target, affected)
            backend.verify(affected)
        except Exception:
            recover(state, backend, journal)
            raise
        atomic_json(
            state / "state.json", {"good": target.model_dump(), "pending": None}
        )
        fence(state, [])


def download_release(run_id: int, destination: Path) -> Release:
    result = subprocess.run(
        ["gh", "api", f"repos/ghalrym/Kadan/actions/runs/{run_id}"],
        check=True,
        capture_output=True,
        text=True,
    )
    run = json.loads(result.stdout)
    if (
        run["path"] != ".github/workflows/compose-release.yml"
        or run["head_branch"] != "master"
        or run["event"] != "push"
        or run["status"] != "completed"
        or run["conclusion"] != "success"
        or run["repository"]["full_name"].lower() != "ghalrym/kadan"
    ):
        raise ValueError(
            "Release must come from the successful master publication workflow"
        )
    subprocess.run(
        [
            "gh",
            "run",
            "download",
            str(run_id),
            "--repo",
            "ghalrym/Kadan",
            "--name",
            "compose-release",
            "--dir",
            str(destination),
        ],
        check=True,
    )
    release = Release.read(destination / "release.json")
    if release.source != run["head_sha"]:
        raise ValueError("Release source does not match trusted publication run")
    return release


@app.command()
def apply(run_id: int, state: Path, checkout: Path):
    """Consume a trusted publication run on an already enrolled installation."""
    with TemporaryDirectory(prefix="kadan-release-") as directory:
        target = download_release(run_id, Path(directory))
        rollout(target, state, Compose(checkout, state))
