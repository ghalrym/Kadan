"""Build immutable images only after reviewed master has passed every CI workflow."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from bin.rollout import SERVICES, WORKERS, Release, atomic_json, download_release

REPOSITORY = "ghalrym/Kadan"
WORKFLOWS = {
    ".github/workflows/api-tests.yml",
    ".github/workflows/services.yml",
    ".github/workflows/inference-worker-tests.yml",
}


def github(path: str):
    return json.loads(
        subprocess.run(
            ["gh", "api", f"repos/{REPOSITORY}/{path}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )


def approved(source: str) -> bool:
    prs = github(f"commits/{source}/pulls")
    for pr in prs:
        if (
            not pr["merged_at"]
            or pr["base"]["ref"] != "master"
            or pr["merge_commit_sha"] != source
        ):
            continue
        reviews = github(f"pulls/{pr['number']}/reviews?per_page=100")
        if len(reviews) >= 100:
            raise ValueError(
                "Review history exceeds verified page; manual publication required"
            )
        latest = {}
        for review in reviews:
            if review["state"] in {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}:
                latest[review["user"]["login"]] = review
        for login, review in latest.items():
            if (
                review["state"] != "APPROVED"
                or review["commit_id"] != pr["head"]["sha"]
                or login == pr["user"]["login"]
            ):
                continue
            permission = github(f"collaborators/{login}/permission")["permission"]
            if permission in {"admin", "maintain", "write"} and not any(
                value["state"] == "CHANGES_REQUESTED" for value in latest.values()
            ):
                return True
    return False


def require_ci(source: str) -> None:
    runs = github(
        f"actions/runs?head_sha={source}&branch=master&event=push&per_page=100"
    )["workflow_runs"]
    # Only the latest attempt for each fixed workflow can admit publication.
    latest = {}
    for run in sorted(
        runs,
        key=lambda value: (value["run_number"], value["run_attempt"]),
        reverse=True,
    ):
        if (
            run["head_sha"] == source
            and run["head_branch"] == "master"
            and run["event"] == "push"
            and run["repository"]["full_name"].lower() == "ghalrym/kadan"
        ):
            latest.setdefault(run["path"], run)
    if any(
        path not in latest
        or latest[path]["status"] != "completed"
        or latest[path]["conclusion"] != "success"
        for path in WORKFLOWS
    ):
        raise ValueError("Exact master commit has not passed all required workflows")


def affected_services(paths: list[str]) -> set[str]:
    affected = set()
    for path in paths:
        if path.startswith("services/tracing/"):
            affected.update(("kadan-api", "kadan-runner", "tsuchi-api", "hikari-api"))
        for service, root in (
            ("kadan", "services/kadan"),
            ("tsuchi", "services/tsuchi"),
        ):
            if path.startswith(root + "/web/"):
                affected.add(service + "-frontend")
            elif path.startswith(root + "/"):
                affected.add(service + "-api")
                if service == "kadan":
                    affected.add("kadan-runner")
        if path.startswith("services/hikari/frontend/"):
            affected.add("hikari-frontend")
        elif path.startswith("services/hikari/api/"):
            affected.add("hikari-api")
        elif path.startswith("services/hikari/"):
            affected.update(("hikari-api", *WORKERS))
        if path == "compose.yaml":
            affected.update(SERVICES)
    return affected


def publish(
    checkout: Path, previous: Release | None, destination: Path, source: str
) -> Release:
    if github("branches/master")["commit"]["sha"] != source or not approved(source):
        raise ValueError("Publication requires the reviewed current master commit")
    require_ci(source)
    if (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        != source
    ):
        raise ValueError("Build checkout does not match verified master SHA")
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip():
        raise ValueError("Publication requires a clean checkout")
    affected = set(SERVICES)
    if previous:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", previous.source, source],
            cwd=checkout,
            check=True,
        )
        paths = subprocess.run(
            ["git", "diff", "--name-only", previous.source, source],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        affected = affected_services(paths)
    images = dict(previous.images) if previous else {}
    tags = {name: f"ghcr.io/ghalrym/kadan/{name}:{source}" for name in SERVICES}
    tags["kadan-runner"] = tags["kadan-api"]
    tags["hikari-migrate"] = tags["hikari-api"]
    overlay = destination.parent / "build-images.json"
    atomic_json(
        overlay, {"services": {name: {"image": image} for name, image in tags.items()}}
    )
    builders = affected - {"kadan-runner"}
    for name in sorted(builders):
        subprocess.run(
            [
                "docker",
                "compose",
                "--env-file",
                "/dev/null",
                "-f",
                "compose.yaml",
                "-f",
                str(overlay),
                "build",
                "--push",
                name,
            ],
            cwd=checkout,
            check=True,
            env={
                **{
                    key: value
                    for key, value in os.environ.items()
                    if key in {"PATH", "HOME", "DOCKER_CONFIG"}
                },
                "HIKARI_WORKER_TOKEN": "build-placeholder-not-a-secret",
            },
        )
        digest = subprocess.run(
            [
                "docker",
                "buildx",
                "imagetools",
                "inspect",
                tags[name],
                "--format",
                "{{json .Manifest.Digest}}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        images[name] = tags[name].rsplit(":", 1)[0] + "@" + json.loads(digest)
    images["kadan-runner"] = images["kadan-api"]
    release = Release(
        source=source,
        compose_sha256=hashlib.sha256(
            (checkout / "compose.yaml").read_bytes()
        ).hexdigest(),
        compatibility="1",
        images=images,
    )
    atomic_json(destination, release.model_dump())
    return release


def build_release(source: str, destination: Path) -> None:
    """Called only by the inactive publication workflow template."""
    deadline = time.monotonic() + 1800
    while True:
        try:
            require_ci(source)
            break
        except ValueError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(30)
    runs = github(
        "actions/workflows/compose-release.yml/runs?branch=master&event=push&status=success&per_page=100"
    )["workflow_runs"]
    previous = None
    if runs:
        baseline = destination.parent / "previous"
        baseline.mkdir()
        previous = download_release(runs[0]["id"], baseline)
    publish(Path.cwd(), previous, destination, source)
