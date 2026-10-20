"""Disposable Compose/recovery primitives using one genuine cached image.

This deliberately does not validate changed-image publication or native inference.
The pending journal is a prepared same-pin recovery fixture, not a published rollout.
"""

import hashlib
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from uuid import uuid4

import typer

from bin import rollout

SCOPE = "CPU protocol stubs; prepared same-pin recovery primitives; no publication or native inference proof"
API_PIN = re.compile(r"ghcr\.io/ghalrym/kadan/kadan-api@sha256:[a-f0-9]{64}\Z")
TIMEOUT = 45


def cached_image(image: str) -> None:
    if not API_PIN.fullmatch(image):
        raise ValueError(
            "Provide a genuine cached GHCR Kadan API sha256 digest; this verifier never pulls or builds images"
        )
    try:
        result = subprocess.run(
            [
                "docker",
                "--host",
                "unix:///var/run/docker.sock",
                "image",
                "inspect",
                image,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(
            "The genuine GHCR API digest must already be cached and Docker access authorized; no resources were created"
        ) from error
    records = json.loads(result.stdout)
    if len(records) != 1 or image not in records[0].get("RepoDigests", []):
        raise ValueError("Docker did not attest the requested cached repository digest")


def docker(*arguments: str) -> str:
    return subprocess.run(
        ["docker", "--host", "unix:///var/run/docker.sock", *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout


def existing_resources(project: str):
    for kind, name in (
        ("volume", project + "-verification"),
        ("network", project + "_default"),
    ):
        names = docker(
            kind, "ls", "--format", "{{.Name}}", "--filter", "name=" + name
        ).splitlines()
        if name in names:
            yield kind, name


def preflight_resources(project: str) -> None:
    if next(existing_resources(project), None) is not None:
        raise ValueError(
            "Disposable volume or network identity already exists; nothing may be adopted"
        )


def assert_owned(backend: rollout.Compose) -> None:
    ids = backend.command("ps", "--all", "--quiet").split()
    if ids:
        records = json.loads(docker("inspect", *ids))
        if {record["Id"] for record in records} != set(ids) or any(
            record["Config"].get("Labels", {}).get("com.docker.compose.project")
            != backend.project
            for record in records
        ):
            raise ValueError(
                "Disposable containers are not owned by this project; cleanup refused"
            )
    for kind, name in existing_resources(backend.project):
        records = json.loads(docker(kind, "inspect", name))
        if (
            len(records) != 1
            or records[0].get("Labels", {}).get("com.docker.compose.project")
            != backend.project
        ):
            raise ValueError("Disposable volume or network is foreign; cleanup refused")


def wait_for(predicate, description: str, timeout: float = TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError(description)
        time.sleep(0.1)


@dataclass
class DisposableInstallation:
    backend: rollout.Compose
    release: rollout.Release
    control: Path
    sentinel: str
    instruction: str

    @classmethod
    def create(cls, directory: Path, image: str):
        checkout, state, host, fixture = (
            directory / name for name in ("checkout", "state", "host", "fixture")
        )
        for path in (
            checkout,
            state,
            host,
            fixture,
            state / "gate",
            fixture / "python" / "kadan_tracing",
            fixture / "bin",
            host / "data",
            host / "control",
        ):
            path.mkdir(parents=True, exist_ok=True)
        project = "kadan-verify-" + uuid4().hex
        sentinel = uuid4().hex
        instruction = "Disposable host instruction " + uuid4().hex
        (host / "AGENT.md").write_text(instruction)
        (host / "data" / "sentinel").write_text(sentinel)
        package = fixture / "python" / "kadan_tracing"
        (package / "__init__.py").write_text("")
        shutil.copyfile(
            Path(__file__).resolve().parents[2]
            / "tracing"
            / "kadan_tracing"
            / "rollout.py",
            package / "rollout.py",
        )
        shutil.copyfile(
            Path(__file__).with_name("fixtures") / "rollout_service.py",
            fixture / "service.py",
        )
        wget = fixture / "bin" / "wget"
        wget.write_text(
            '#!/usr/local/bin/python\nimport os,sys\nos.execv(sys.executable,[sys.executable,"/fixture/service.py","fetch",sys.argv[-1]])\n'
        )
        wget.chmod(0o755)
        services = {}
        for service in rollout.SERVICES:
            services[service] = {
                "image": image,
                "entrypoint": ["python"],
                "command": ["/fixture/service.py", "serve"],
                "user": "0:0",
                "restart": "no",
                "init": True,
                "read_only": True,
                "cap_drop": ["ALL"],
                "security_opt": ["no-new-privileges:true"],
                "cpus": 0.1,
                "mem_limit": "64m",
                "pids_limit": 32,
                "tmpfs": ["/tmp"],
                "environment": {
                    "FIXTURE_SERVICE": service,
                    "FIXTURE_SENTINEL": sentinel,
                    "PYTHONPATH": "/fixture/python",
                    "PATH": "/fixture/bin:/usr/local/bin:/usr/bin:/bin",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                "volumes": [
                    {
                        "type": "bind",
                        "source": str(fixture),
                        "target": "/fixture",
                        "read_only": True,
                    },
                    {
                        "type": "bind",
                        "source": str(host / "control"),
                        "target": "/control",
                        "read_only": True,
                    },
                    {
                        "type": "volume",
                        "source": "verification",
                        "target": "/verification",
                    },
                ],
                "healthcheck": {
                    # Avoid importing the server in a probe under the CPU quota.
                    "test": [
                        "CMD",
                        "python",
                        "-c",
                        "import os; raise SystemExit(0 if "
                        "os.path.exists('/tmp/fixture-ready') and not "
                        "os.path.exists('/control/unhealthy-' + "
                        "os.environ['FIXTURE_SERVICE']) else 1)",
                    ],
                    "interval": "1s",
                    "timeout": "2s",
                    "retries": 3,
                    "start_period": "10s",
                },
            }
        topology = {"services": services, "volumes": {"verification": {}}}
        (checkout / "compose.yaml").write_text(json.dumps(topology))
        (checkout / ".env").write_text("")
        mounts = {
            name: {
                "volumes": [
                    {
                        "type": "bind",
                        "source": str(host / "AGENT.md"),
                        "target": "/app/AGENT.md",
                        "read_only": True,
                    }
                ]
            }
            for name in ("kadan-api", "kadan-runner")
        }
        mounts["tsuchi-api"] = {
            "volumes": [
                {"type": "bind", "source": str(host / "data"), "target": "/data"}
            ]
        }
        overlay = host / "overlay.json"
        overlay.write_text(
            json.dumps(
                {
                    "services": mounts,
                    "volumes": {"verification": {"name": project + "-verification"}},
                }
            )
        )
        installation = {
            "project": project,
            "supervision": "detached-independent",
            "env_sha256": hashlib.sha256(b"").hexdigest(),
            "host_overlay": {
                "path": str(overlay),
                "sha256": hashlib.sha256(overlay.read_bytes()).hexdigest(),
                "content_pins": [
                    {
                        "path": str(host / "AGENT.md"),
                        "sha256": hashlib.sha256(instruction.encode()).hexdigest(),
                    }
                ],
            },
        }
        rollout.atomic_json(state / "installation.json", installation)
        source = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        release = rollout.Release(
            source=source,
            compose_sha256=hashlib.sha256(
                (checkout / "compose.yaml").read_bytes()
            ).hexdigest(),
            compatibility="disposable-protocol-fixture",
            images=dict.fromkeys(rollout.SERVICES, image),
        )
        rollout.atomic_json(
            state / "state.json", {"good": release.model_dump(), "pending": None}
        )
        rollout.fence(state, [])
        backend = rollout.Compose(checkout, state, timeout=TIMEOUT)
        backend.overlay(release)
        return cls(backend, release, host / "control", sentinel, instruction)

    def preservation(self):
        expected = {"volume": hashlib.sha256(self.sentinel.encode()).hexdigest()}
        for service in ("kadan-api", "kadan-runner", "tsuchi-api"):
            observed = json.loads(
                self.backend.command(
                    "exec", "-T", service, "python", "/fixture/service.py", "inspect"
                )
            )
            values = {
                **expected,
                "bind" if service == "tsuchi-api" else "instruction": hashlib.sha256(
                    (
                        self.sentinel if service == "tsuchi-api" else self.instruction
                    ).encode()
                ).hexdigest(),
            }
            if any(observed.get(key) != value for key, value in values.items()):
                raise AssertionError("Host instruction or persistent data changed")

    def drain(self):
        hold = self.control / "hold-kadan-api"
        hold.touch()
        try:
            wait_for(
                lambda: self.backend.status("kadan-api")["active"] == 1,
                "Fixture work was not admitted",
            )
            before = self.backend.snapshot_epochs()
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(self.backend.drain)
                try:
                    wait_for(
                        lambda: self.backend.status("kadan-api")["fenced"],
                        "Fixture admission was not fenced",
                    )
                    if pending.done():
                        raise AssertionError("Drain passed accepted work")
                    self.backend.unchanged_epochs(before)
                finally:
                    hold.unlink(missing_ok=True)
                pending.result(timeout=TIMEOUT)
        finally:
            hold.unlink(missing_ok=True)

    def recovery(self):
        backend = self.backend
        baseline = backend.snapshot_epochs()
        # Prepared same-pin journal: exercises recovery, not a fictional publication.
        journal = {
            "previous": self.release.model_dump(),
            "target": self.release.model_dump(),
            "affected": ["kadan-api", "kadan-runner"],
            "phase": "replacing",
            "native": None,
            "runner": backend.runner_transition(),
            "baseline": baseline,
        }
        rollout.save_pending(backend.state, journal)
        backend.handoff_runner(journal)
        observed, lease = backend.lease_snapshot(journal["runner"]["runner_id"])
        if lease.lease_expires_at > observed:
            raise AssertionError(
                "Runner replaced before its observed natural lease expiry"
            )
        backend.replace(self.release, tuple(journal["affected"]))
        backend.verify(tuple(journal["affected"]))
        backend.verify_runner(journal, self.release)
        self.preservation()
        fault = self.control / "unhealthy-kadan-api"
        fault.touch()
        try:
            try:
                backend.verify(tuple(journal["affected"]))
            except TimeoutError:
                pass
            else:
                raise AssertionError(
                    "Unhealthy fixture passed replacement verification"
                )
        finally:
            fault.unlink(missing_ok=True)
        context = multiprocessing.get_context("spawn")
        process = context.Process(
            target=checkpoint_and_exit, args=(backend.state, journal)
        )
        process.start()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)
            raise TimeoutError("Checkpoint fixture did not exit")
        if process.exitcode != 17:
            raise AssertionError("Expected intentional fixture controller interruption")
        with rollout.locked(backend.state):
            rollout.resume_locked(backend.state, backend)
        record = json.loads((backend.state / "state.json").read_text())
        if record["pending"] is not None or record["good"] != self.release.model_dump():
            raise AssertionError("Interrupted same-pin recovery did not finalize")
        self.preservation()


def checkpoint_and_exit(state: Path, journal: dict):
    with rollout.locked(state):
        rollout.save_pending(state, journal)
        os._exit(17)


def lock_waiter(state: Path, acquired):
    with rollout.locked(state):
        acquired.set()


def serialization(state: Path):
    context = multiprocessing.get_context("spawn")
    acquired = context.Event()
    process = context.Process(target=lock_waiter, args=(state, acquired))
    try:
        with rollout.locked(state):
            process.start()
            if acquired.wait(0.2):
                raise AssertionError("Another controller passed the deployment lock")
        if not acquired.wait(10):
            raise TimeoutError("Deployment lock was not released")
        process.join(5)
        if process.exitcode != 0:
            raise AssertionError("Serialized fixture controller failed")
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)


def verify(image: str, report: Path) -> dict:
    cached_image(image)
    directory = Path(mkdtemp(prefix="kadan-verify-"))
    result = {
        "scope": SCOPE,
        "image": image,
        "checks": [],
        "cleaned": False,
        "passed": False,
        "workdir": str(directory),
    }
    fixture = None
    created = False
    try:
        fixture = DisposableInstallation.create(directory, image)
        backend = fixture.backend
        result["project"] = backend.project
        if backend.command("ps", "--all", "--quiet").strip():
            raise ValueError("Disposable project identity already exists")
        preflight_resources(backend.project)
        created = True
        # Create without starting: refuse adopted resources before fixture processes
        # can access data, even if another actor raced the collision checks.
        backend.command("create", "--no-build", "--pull", "never", *rollout.SERVICES)
        assert_owned(backend)
        backend.command(
            "up",
            "-d",
            "--no-build",
            "--pull",
            "never",
            "--wait",
            "--wait-timeout",
            "30",
            *rollout.SERVICES,
        )
        backend.assert_running(fixture.release)
        fixture.preservation()
        result["checks"].append(
            "host instruction, bind data and named volume preserved"
        )
        fixture.drain()
        result["checks"].append(
            "accepted work blocks drain without stopping containers"
        )
        fixture.recovery()
        result["checks"].append(
            "natural lease expiry, health failure and interrupted same-pin recovery"
        )
        serialization(backend.state)
        result["checks"].append(
            "controller serialization and lock release after interruption"
        )
        result["passed"] = True
    except Exception as error:
        result["error"] = type(error).__name__
        raise
    finally:
        try:
            if created:
                backend = fixture.backend
                assert_owned(backend)
                backend.command("down", "--volumes", "--remove-orphans")
                if (
                    backend.command("ps", "--all", "--quiet").strip()
                    or next(existing_resources(backend.project), None) is not None
                ):
                    raise RuntimeError("Disposable resources remain after cleanup")
            result["cleaned"] = True
        except Exception as error:
            result["passed"] = False
            result["cleanup_error"] = type(error).__name__
            raise
        finally:
            report.parent.mkdir(parents=True, exist_ok=True)
            rollout.atomic_json(report, result)
            if result["cleaned"]:
                shutil.rmtree(directory)
    return result


def run(image: str, report: Path):
    """Verify disposable CPU recovery using an already cached genuine GHCR API pin."""
    typer.echo(json.dumps(verify(image, report), indent=2))
