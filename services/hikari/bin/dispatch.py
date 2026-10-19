"""Host-local polling of the one trusted publication workflow; never installed here."""

import json
import logging
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from bin.rollout import (
    SHA,
    Compose,
    atomic_json,
    download_release,
    locked,
    resume_locked,
    rollout_locked,
)

logger = logging.getLogger(__name__)


class DiscoveryError(RuntimeError):
    """A read-only remote lookup failed before a deployment attempt."""


def successful_runs() -> list[dict]:
    result = subprocess.run(
        [
            "gh",
            "api",
            "--paginate",
            "--slurp",
            (
                "repos/ghalrym/Kadan/actions/workflows/compose-release.yml/runs"
                "?branch=master&event=push&status=success&per_page=100"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    values = [
        run for page in json.loads(result.stdout) for run in page["workflow_runs"]
    ]
    for run in values:
        if (
            run["path"] != ".github/workflows/compose-release.yml"
            or run["head_branch"] != "master"
            or run["event"] != "push"
            or run["status"] != "completed"
            or run["conclusion"] != "success"
            or run["repository"]["full_name"].lower() != "ghalrym/kadan"
            or not SHA.fullmatch(run["head_sha"])
        ):
            raise ValueError("Unexpected publication provenance in dispatcher poll")
    return sorted(
        values, key=lambda run: (run["run_number"], run["run_attempt"]), reverse=True
    )


def ancestor(checkout: Path, before: str, after: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", before, after],
        cwd=checkout,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError("Cannot determine release ancestry")
    return result.returncode == 0


def poll_once(state: Path, checkout: Path, backend: Compose) -> int | None:
    with locked(state):
        # Recover before asking the network for another release. The cursor shares
        # this lock with apply, so two dispatchers cannot advance it concurrently.
        current = resume_locked(state, backend)
        cursor_path = state / "dispatcher.json"
        cursor = (
            json.loads(cursor_path.read_text())
            if cursor_path.exists()
            else {"processed": []}
        )
        processed = list(cursor["processed"])
        try:
            runs = successful_runs()
        except (OSError, subprocess.SubprocessError) as error:
            raise DiscoveryError("Publication discovery unavailable") from error
        for run in runs:
            delivery = f"{run['id']}:{run['run_attempt']}"
            if delivery in processed:
                continue
            try:
                subprocess.run(
                    [
                        "git",
                        "fetch",
                        "--no-tags",
                        "https://github.com/ghalrym/Kadan.git",
                        run["head_sha"],
                    ],
                    cwd=checkout,
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise DiscoveryError("Publication source lookup unavailable") from error
            if run["head_sha"] != current.source and ancestor(
                checkout, run["head_sha"], current.source
            ):
                # Trusted successful workflow metadata plus source ancestry proves
                # this delivery cannot advance the installation. Continue scanning;
                # a lower-number run may have completed late with a descendant.
                processed.append(delivery)
                atomic_json(
                    cursor_path,
                    {
                        "processed": processed[-1000:],
                        "run_id": run["id"],
                        "installed_source": current.source,
                    },
                )
                logger.info("Skipping proven older publication run %s", run["id"])
                continue
            with TemporaryDirectory(prefix="kadan-dispatch-") as directory:
                try:
                    target = download_release(run["id"], Path(directory))
                except (OSError, subprocess.SubprocessError) as error:
                    raise DiscoveryError(
                        "Publication artifact lookup unavailable"
                    ) from error
                rollout_locked(target, state, backend)
            processed.append(delivery)
            atomic_json(
                cursor_path,
                {
                    "processed": processed[-1000:],
                    "run_id": run["id"],
                    "installed_source": target.source,
                },
            )
            return run["id"]
        return None


def watch(state: Path, checkout: Path):
    """Poll trusted successful releases every minute on an enrolled local host."""
    backend = Compose(checkout, state)
    delay = 60
    while True:
        try:
            poll_once(state, checkout, backend)
            delay = 60
        except DiscoveryError:
            logger.exception("Publication discovery failed; cursor not advanced")
            delay = min(delay * 2, 300)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
            logger.exception("Rollout dispatcher paused; operator attention required")
            raise
        # KeyboardInterrupt/SIGTERM do not invoke rollback while shutdown is in
        # progress. A durable pending journal is recovered by the next invocation.
        time.sleep(delay)
