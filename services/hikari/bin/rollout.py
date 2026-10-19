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
from uuid import UUID, uuid4

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
NATIVE = ("hikari-api", *WORKERS)
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


class NativeWorker(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: str
    instance: str
    generation: UUID


class NativeProof(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    native_idle: bool
    api_generation: UUID
    cold_start_generation: str | None
    workers: tuple[NativeWorker, ...]

    @model_validator(mode="after")
    def valid_idle_cohort(self):
        expected = {name.removeprefix("hikari-worker-") for name in WORKERS}
        if (
            not self.native_idle
            or len(self.workers) != len(WORKERS)
            or {worker.kind for worker in self.workers} != expected
            or len({worker.generation for worker in self.workers}) != len(WORKERS)
        ):
            raise ValueError("Native barrier did not prove one complete idle cohort")
        return self


class ContainerIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    started: str
    image: str
    hostname: str
    generation: str | None


def affected_services(current: Release, target: Release) -> tuple[str, ...]:
    affected = {
        name for name in SERVICES if current.images[name] != target.images[name]
    }
    if affected.intersection(NATIVE):
        affected.update(NATIVE)
    return tuple(name for name in SERVICES if name in affected)


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
        marker = self.state / "gate" / "native.json"
        if marker.exists():
            generation = json.loads(marker.read_text())["generation"]
            for name in NATIVE:
                services[name].setdefault("environment", {})[
                    "KADAN_ROLLOUT_GENERATION"
                ] = generation
        atomic_json(self.state / "images.json", {"services": services})

    def preflight(self, current: Release, target: Release) -> None:
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

    def cached_images(self, release: Release, services: tuple[str, ...]) -> None:
        references = sorted({release.images[name] for name in services})
        subprocess.run(
            [
                "docker",
                "--host",
                "unix:///var/run/docker.sock",
                "image",
                "inspect",
                *references,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

    def prepare_images(
        self, current: Release, target: Release, services: tuple[str, ...]
    ) -> None:
        # A running container does not prove its digest reference remains cached.
        self.cached_images(current, services)
        self.overlay(target)
        try:
            self.command("pull", *services)
            self.cached_images(target, services)
        finally:
            self.overlay(current)

    def replace(self, release: Release, services: tuple[str, ...]) -> None:
        self.overlay(release)
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

    def containers(self, services: tuple[str, ...]) -> dict[str, ContainerIdentity]:
        ids = self.command("ps", "--all", "--quiet", *services).split()
        if not ids:
            return {}
        values = json.loads(
            subprocess.run(
                ["docker", "--host", "unix:///var/run/docker.sock", "inspect", *ids],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        result = {}
        for value in values:
            name = value["Config"]["Labels"]["com.docker.compose.service"]
            if name not in services or name in result:
                raise ValueError("Native cohort has unknown services or replicas")
            environment = dict(
                item.split("=", 1) for item in value["Config"]["Env"] if "=" in item
            )
            result[name] = ContainerIdentity(
                id=value["Id"],
                started=value["State"]["StartedAt"],
                image=value["Config"]["Image"],
                hostname=value["Config"]["Hostname"],
                generation=environment.get("KADAN_ROLLOUT_GENERATION"),
            )
        return result

    def persisted_old(self, native: dict) -> dict[str, dict]:
        observed = {}
        for name, identity in native["old"].items():
            old = ContainerIdentity(**identity)
            # A successful daemon listing with this exact full ID proves removal;
            # daemon/permission errors never mean that an old process is absent.
            visible = subprocess.run(
                [
                    "docker",
                    "--host",
                    "unix:///var/run/docker.sock",
                    "ps",
                    "--all",
                    "--quiet",
                    "--no-trunc",
                    "--filter",
                    "id=" + old.id,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            ).stdout.split()
            if not visible:
                continue
            if visible != [old.id]:
                raise ValueError("Ambiguous persisted native container ID")
            values = json.loads(
                subprocess.run(
                    [
                        "docker",
                        "--host",
                        "unix:///var/run/docker.sock",
                        "inspect",
                        old.id,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                ).stdout
            )
            if len(values) != 1 or values[0]["Id"] != old.id:
                raise ValueError("Persisted native container observation changed")
            value = values[0]
            if (
                value["State"]["StartedAt"] != old.started
                or value["Config"]["Image"] != old.image
                or value["Config"]["Hostname"] != old.hostname
            ):
                raise ValueError("Persisted native process epoch changed; keep fence")
            observed[name] = value
        return observed

    def barrier(self) -> NativeProof:
        return NativeProof.model_validate_json(
            self.command(
                "exec",
                "-T",
                "hikari-api",
                "python",
                "-m",
                "kadan_tracing.rollout",
                "barrier",
            )
        )

    def native_transition(self, previous: Release) -> dict:
        before = self.containers(NATIVE)
        if set(before) != set(NATIVE):
            raise ValueError("Native cohort is incomplete before its idle barrier")
        proof = self.barrier()
        after = self.containers(NATIVE)
        if before != after:
            raise ValueError("Native container identity changed during idle proof")
        for worker in proof.workers:
            container = after["hikari-worker-" + worker.kind]
            if worker.instance != container.hostname:
                raise ValueError(
                    "Native proof does not match enrolled container identity"
                )
        if any(
            container.image != previous.images[name]
            for name, container in after.items()
        ):
            raise ValueError("Native images changed during idle proof")
        return {
            "proof": proof.model_dump(mode="json"),
            "old": {name: value.model_dump() for name, value in after.items()},
            "generations": [],
            "fresh_proof": None,
            "stop_authorized": False,
        }

    def replace_native(self, release: Release, journal: dict) -> None:
        closed = json.loads((self.state / "gate" / "fence.json").read_text())["closed"]
        if set(closed) != set(OWNERS):
            raise ValueError(
                "Native replacement requires the complete persistent fence"
            )
        native = journal["native"]
        proof = NativeProof(**native["proof"])
        old = {
            name: ContainerIdentity(**value) for name, value in native["old"].items()
        }
        if set(old) != set(NATIVE):
            raise ValueError("Persisted native idle proof is incomplete")
        previous, target = Release(**journal["previous"]), Release(**journal["target"])
        cohort = self.containers(NATIVE)
        persisted = self.persisted_old(native)
        for name, value in persisted.items():
            if value["State"]["Running"] and (
                name not in cohort or cohort[name].id != value["Id"]
            ):
                raise ValueError(
                    "An old native process is hidden from the enrolled cohort"
                )
        for name, container in cohort.items():
            original = (
                container.id == old[name].id and container.started == old[name].started
            )
            replacement = container.generation in native[
                "generations"
            ] and container.image in {previous.images[name], target.images[name]}
            if not (original or replacement):
                raise ValueError(
                    "Unknown native process epoch; preserve fence for manual recovery"
                )
        originals = all(
            name in cohort
            and cohort[name].id == old[name].id
            and cohort[name].started == old[name].started
            for name in NATIVE
        )
        if not native["stop_authorized"] or (
            originals and all(not self.stopped(name) for name in NATIVE)
        ):
            live = self.barrier()
            if live != proof:
                raise ValueError(
                    "Native idle receipt changed before stopping; keep fence"
                )
            native["stop_authorized"] = True
            save_pending(self.state, journal)
        # Every old epoch was proven idle; every new epoch was born under this
        # transaction's still-closed fence and has never accepted application work.
        self.stop_native_ids(
            tuple(cohort[name].id for name in WORKERS if name in cohort)
        )
        if "hikari-api" in cohort:
            self.stop_native_ids((cohort["hikari-api"].id,))
        if self.containers(NATIVE) != cohort:
            raise RuntimeError("Native container identity changed while stopping")
        if any(
            value["State"]["Running"] for value in self.persisted_old(native).values()
        ):
            raise RuntimeError("A persisted old native process remains running")
        generation = str(uuid4())
        native["generations"].append(generation)
        journal["phase"] = "native_stopped"
        save_pending(self.state, journal)
        atomic_json(
            self.state / "gate" / "native.json",
            {
                "phase": "cold_start",
                "generation": generation,
                "old_workers": [
                    value.model_dump(mode="json") for value in proof.workers
                ],
            },
        )
        self.overlay(release)
        self.command(
            "up",
            "-d",
            "--no-deps",
            "--no-build",
            "--pull",
            "never",
            "--force-recreate",
            "--wait",
            "--wait-timeout",
            str(self.timeout),
            "hikari-api",
        )
        self.command(
            "up",
            "-d",
            "--no-deps",
            "--no-build",
            "--pull",
            "never",
            "--force-recreate",
            *WORKERS,
        )
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fresh = self.barrier()
                if (
                    fresh.cold_start_generation != generation
                    or fresh.api_generation == proof.api_generation
                ):
                    raise ValueError(
                        "Cold-start barrier does not match the controller generation"
                    )
                if {worker.generation for worker in fresh.workers}.intersection(
                    worker.generation for worker in proof.workers
                ):
                    raise ValueError(
                        "A previous native process reconnected after replacement"
                    )
                containers = self.containers(NATIVE)
                if set(containers) != set(NATIVE) or any(
                    value.generation != generation
                    or value.id == old[name].id
                    or value.image != release.images[name]
                    for name, value in containers.items()
                ):
                    raise ValueError(
                        "Cold-start container cohort is not fresh and pinned"
                    )
                if any(
                    worker.instance
                    != containers["hikari-worker-" + worker.kind].hostname
                    for worker in fresh.workers
                ):
                    raise ValueError("Cold-start native identity mismatch")
                native["fresh_proof"] = fresh.model_dump(mode="json")
                journal["phase"] = "replacing"
                save_pending(self.state, journal)
                return
            except subprocess.CalledProcessError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Fresh native cohort did not prove idle"
                    ) from None
                time.sleep(0.2)

    def stop_native_ids(self, ids: tuple[str, ...]) -> None:
        if not ids:
            return
        subprocess.run(
            [
                "docker",
                "--host",
                "unix:///var/run/docker.sock",
                "stop",
                "--time",
                "30",
                *ids,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        values = json.loads(
            subprocess.run(
                ["docker", "--host", "unix:///var/run/docker.sock", "inspect", *ids],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            ).stdout
        )
        if {value["Id"] for value in values} != set(ids) or any(
            value["State"]["Running"]
            or value["State"]["Status"] not in {"exited", "dead", "created"}
            for value in values
        ):
            raise RuntimeError("Native container IDs are not conclusively stopped")

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


def save_pending(state: Path, journal: dict) -> None:
    atomic_json(state / "state.json", {"good": journal["previous"], "pending": journal})


def recover(state: Path, backend: Compose, journal: dict) -> None:
    if journal["phase"] not in {
        "draining",
        "native_stopping",
        "native_stopped",
        "replacing",
    }:
        raise ValueError("Unknown recovery phase; keep admission fenced")
    previous = Release(**journal["previous"])
    logger.warning("Recovering release %s phase %s", previous.source, journal["phase"])
    if journal["phase"] != "draining":
        backend.cached_images(previous, tuple(journal["affected"]))
        if journal.get("native"):
            backend.replace_native(previous, journal)
            others = tuple(name for name in journal["affected"] if name not in NATIVE)
            if others:
                backend.stop_failed_replacements({**journal, "affected": others})
                backend.drain(others)
                backend.replace(previous, others)
        else:
            backend.stop_failed_replacements(journal)
            backend.drain(tuple(journal["affected"]))
            backend.replace(previous, tuple(journal["affected"]))
        backend.verify(tuple(journal["affected"]))
    atomic_json(state / "state.json", {"good": previous.model_dump(), "pending": None})
    fence(state, [])


def resume_locked(state: Path, backend: Compose) -> Release:
    record = json.loads((state / "state.json").read_text())
    backend.configuration(Release(**record["good"]))
    if record["pending"]:
        recover(state, backend, record["pending"])
        record = json.loads((state / "state.json").read_text())
    current = Release(**record["good"])
    gate = state / "gate" / "fence.json"
    if gate.exists() and json.loads(gate.read_text())["closed"]:
        backend.assert_running(current)
        backend.drain()
        backend.verify(SERVICES)
        fence(state, [])
    return current


def rollout_locked(target: Release, state: Path, backend: Compose) -> None:
    current = resume_locked(state, backend)
    backend.preflight(current, target)
    if current == target:
        return
    affected = affected_services(current, target)
    if not affected:
        atomic_json(
            state / "state.json", {"good": target.model_dump(), "pending": None}
        )
        return
    backend.prepare_images(current, target, affected)
    journal = {
        "previous": current.model_dump(),
        "target": target.model_dump(),
        "affected": affected,
        "phase": "draining",
        "native": None,
    }
    save_pending(state, journal)
    try:
        backend.drain()
        if set(affected).intersection(NATIVE):
            journal["native"] = backend.native_transition(current)
            journal["phase"] = "native_stopping"
        else:
            journal["phase"] = "replacing"
        save_pending(state, journal)
        logger.info(
            "Replacing release %s services %s", target.source, ",".join(affected)
        )
        if journal["native"]:
            backend.replace_native(target, journal)
            others = tuple(name for name in affected if name not in NATIVE)
            if others:
                backend.replace(target, others)
        else:
            backend.replace(target, affected)
        backend.verify(affected)
    except Exception:
        recover(state, backend, journal)
        raise
    atomic_json(state / "state.json", {"good": target.model_dump(), "pending": None})
    fence(state, [])


def rollout(target: Release, state: Path, backend: Compose) -> None:
    with locked(state):
        rollout_locked(target, state, backend)


def download_release(run_id: int, destination: Path) -> Release:
    result = subprocess.run(
        ["gh", "api", f"repos/ghalrym/Kadan/actions/runs/{run_id}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
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
        timeout=120,
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
