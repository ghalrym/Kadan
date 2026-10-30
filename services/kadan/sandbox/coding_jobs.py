"""Bounded Codex execution owned by the shared Kadan computer."""

from __future__ import annotations

import base64
import ctypes
import fcntl
import hashlib
import json
import os
import re
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

STATE = Path("/home/gem/.local/state/kadan/coding")
VERSION = "0.162.1"
CODEX_HOME = "/home/gem/.codex"
ACTIVE = {"starting", "running", "stopping", "cleanup_required"}
TERMINAL = {"completed", "failed", "cancelled", "timed_out", "interrupted"}
MAX_EVENTS = 100


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_start(pid: int) -> str | None:
    try:
        values = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return values[19] if values[0] not in {"Z", "X"} else None
    except (OSError, IndexError):
        return None


def incarnation() -> str:
    return (
        Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        + ":"
        + str(process_start(1))
    )


def final_summary(text: str) -> str:
    # Command output never enters state. Redact common credentials in the final answer too.
    text = re.sub(
        r"(?i)(bearer\s+|(?:api[_-]?key|token|password|secret)\s*[:=]\s*)\S+",
        r"\1[redacted]",
        text,
    )
    text = re.sub(r"\b(?:sk-|gh[pousr]_|github_pat_)[A-Za-z0-9_-]+", "[redacted]", text)
    return "".join(c for c in text if c == "\n" or ord(c) >= 32)[:2000]


@dataclass
class Job:
    id: str
    request_id: str
    fingerprint: str
    cwd: str
    resume_job_id: str | None
    incarnation: str
    created_at: str = field(default_factory=now)
    status: str = "starting"
    session_id: str | None = None
    finished_at: str | None = None
    message: str = "Starting Codex."
    events: list[dict] = field(default_factory=list)
    sequence: int = 0
    worker_pid: int | None = None
    worker_start: str | None = None
    launched_at: float = field(default_factory=time.time)
    timeout_seconds: int = 1800
    cancel_requested: bool = False
    cleanup_confirmed: bool = False

    def event(self, kind: str, message: str) -> None:
        self.sequence += 1
        self.events.append(
            {"sequence": self.sequence, "kind": kind, "message": message[:300]}
        )
        self.events = self.events[-MAX_EVENTS:]

    def public(self, after: int = 0, latest: bool = False) -> dict:
        available = [event for event in self.events if event["sequence"] > after]
        events = available[-20:] if latest else available[:20]
        truncated = bool(events and events[0]["sequence"] > after + 1)
        return {
            key: getattr(self, key)
            for key in (
                "id",
                "request_id",
                "status",
                "cwd",
                "session_id",
                "resume_job_id",
                "created_at",
                "finished_at",
                "message",
            )
        } | {
            "events": events,
            "next_cursor": events[-1]["sequence"]
            if events
            else max(after, self.sequence),
            "events_truncated": truncated,
        }


class Jobs:
    def __init__(self, root: Path = STATE):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    @contextmanager
    def locked(self):
        with (self.root / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def save(self, job: Job) -> None:
        fd, name = tempfile.mkstemp(dir=self.root, prefix=".state-")
        try:
            with os.fdopen(fd, "w") as output:
                json.dump(asdict(job), output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(name, self.root / f"{job.id}.json")
        finally:
            Path(name).unlink(missing_ok=True)

    def get(self, job_id: str) -> Job:
        try:
            return Job(**json.loads((self.root / f"{job_id}.json").read_text()))
        except FileNotFoundError:
            raise ValueError("Coding job was not found.") from None

    def all(self) -> list[Job]:
        return sorted(
            (Job(**json.loads(path.read_text())) for path in self.root.glob("*.json")),
            key=lambda job: job.created_at,
            reverse=True,
        )

    def reconcile(self) -> list[Job]:
        jobs = self.all()
        runtime = incarnation()
        for job in jobs:
            if job.status not in ACTIVE:
                continue
            if job.incarnation != runtime:
                job.status = "interrupted"
                job.message = "Computer restarted; Codex was interrupted. Inspect files before starting another turn."
                job.finished_at = now()
                job.cleanup_confirmed = True
            elif job.worker_pid and process_start(job.worker_pid) == job.worker_start:
                continue
            elif (
                job.status == "starting"
                and not job.worker_pid
                and time.time() - job.launched_at < 15
            ):
                continue
            else:
                job.status = "cleanup_required"
                job.message = "Codex supervisor stopped unexpectedly. Restart the computer to confirm process cleanup before another job."
                job.cleanup_confirmed = False
            (self.root / f"{job.id}.prompt").unlink(missing_ok=True)
            self.save(job)
        return jobs

    def start(self, request: dict) -> dict:
        request_id = valid_uuid(request.get("request_id"))
        prompt = request.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16000:
            raise ValueError("Prompt must contain 1 to 16000 characters.")
        cwd = request.get("cwd")
        if not isinstance(cwd, str) or not Path(cwd).is_absolute():
            raise ValueError(
                "Working directory must be an existing absolute directory in the computer."
            )
        timeout = request.get("timeout_seconds", 1800)
        if type(timeout) is not int or not 60 <= timeout <= 3600:
            raise ValueError("Timeout must be an integer between 60 and 3600 seconds.")
        previous_id = (
            valid_uuid(request["resume_job_id"])
            if request.get("resume_job_id") is not None
            else None
        )
        fingerprint = hashlib.sha256(
            json.dumps([prompt, cwd, timeout, previous_id]).encode()
        ).hexdigest()
        with self.locked():
            jobs = self.reconcile()
            for existing in jobs:
                if existing.request_id == request_id:
                    if existing.fingerprint != fingerprint:
                        raise ValueError(
                            "Request ID was already used with different coding instructions."
                        )
                    return existing.public()
            # Retries compare the original inputs even if earlier work removed or
            # renamed its directory. Runtime validation applies only to new work.
            if not Path(cwd).is_dir():
                raise ValueError(
                    "Working directory must be an existing absolute directory in the computer."
                )
            cwd = str(Path(cwd).resolve())
            if any(job.status in ACTIVE for job in jobs):
                raise ValueError(
                    "The shared computer already has an active coding job or requires cleanup. Inspect coding status first."
                )
            if previous_id:
                previous = self.get(previous_id)
                if (
                    previous.status not in TERMINAL
                    or not previous.cleanup_confirmed
                    or not previous.session_id
                ):
                    raise ValueError(
                        "Resume requires a recorded Codex session with confirmed process cleanup."
                    )
                if previous.cwd != cwd:
                    raise ValueError(
                        "Resume must use the previous coding job's working directory."
                    )
            setup = setup_status()
            if setup["state"] != "ready":
                raise ValueError(setup["message"])
            job = Job(
                id=request_id,
                request_id=request_id,
                fingerprint=fingerprint,
                cwd=cwd,
                resume_job_id=previous_id,
                incarnation=incarnation(),
                timeout_seconds=timeout,
            )
            if previous_id:
                job.session_id = previous.session_id
            job.event(
                "starting",
                "Coding job accepted; it will not be replayed automatically.",
            )
            self.save(job)
            prompt_path = self.root / f"{job.id}.prompt"
            try:
                with open(
                    prompt_path,
                    "x",
                    opener=lambda path, flags: os.open(path, flags, 0o600),
                ) as output:
                    output.write(prompt)
                worker = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "worker", job.id],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    close_fds=True,
                )
                job.worker_pid = worker.pid
                job.worker_start = process_start(worker.pid)
                self.save(job)
            except OSError:
                prompt_path.unlink(missing_ok=True)
                # Popen failed before a child existed. A failure saving a spawned child remains uncertain.
                if job.worker_pid is None:
                    job.status, job.message = (
                        "failed",
                        "Codex supervisor could not start.",
                    )
                    job.finished_at, job.cleanup_confirmed = now(), True
                    self.save(job)
                else:
                    raise
            return job.public()

    def cancel(self, job_id: str) -> dict:
        with self.locked():
            self.reconcile()
            job = self.get(job_id)
            if job.status in {"starting", "running"}:
                job.cancel_requested = True
                job.status, job.message = (
                    "stopping",
                    "Stopping Codex and cleaning up its processes.",
                )
                job.event("stopping", job.message)
                self.save(job)
            return job.public()

    def status(self, job_id: str | None, after: int = 0) -> dict:
        with self.locked():
            jobs = self.reconcile()
            if job_id:
                return self.get(job_id).public(after)
            recent = jobs[:20]
            active = [job for job in jobs if job.status in ACTIVE and job not in recent]
            result = [job.public(latest=True) for job in (active + recent)[:20]]
        return {"setup": setup_status(), "jobs": result}


def valid_uuid(value) -> str:
    try:
        return str(UUID(value))
    except (ValueError, TypeError, AttributeError):
        raise ValueError(
            "A valid UUID is required for coding job and request IDs."
        ) from None


def setup_status() -> dict:
    binary = shutil.which("codex")
    if not binary:
        return {
            "state": "missing",
            "message": "Codex CLI is missing from the computer. Build the configured Kadan computer image.",
        }
    try:
        version = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            timeout=5,
            check=False,
            env={**os.environ, "CODEX_HOME": CODEX_HOME},
        )
        if (
            version.returncode != 0
            or version.stdout.decode(errors="replace").strip() != f"codex-cli {VERSION}"
        ):
            return {
                "state": "unavailable",
                "message": "The computer requires the supported Codex CLI version. Rebuild the configured Kadan computer image.",
            }
        auth = subprocess.run(
            [binary, "login", "status"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
            env={**os.environ, "CODEX_HOME": CODEX_HOME},
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"state": "unavailable", "message": "Codex setup could not be checked."}
    if auth.returncode:
        return {
            "state": "auth_required",
            "version": VERSION,
            "message": "Codex is installed. The owner must sign in inside the computer before coding jobs can start.",
        }
    return {
        "state": "ready",
        "version": VERSION,
        "message": "Codex is installed and reports signed-in credentials. Execution has not been verified.",
    }


def command(job: Job) -> list[str]:
    config = [
        "--json",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "-c",
        'model_reasoning_effort="high"',
        "-c",
        'approval_policy="never"',
        "-m",
        "gpt-6.1-sol",
    ]
    if job.resume_job_id:
        return [
            "codex",
            "exec",
            "resume",
            *config,
            "-c",
            'sandbox_mode="workspace-write"',
            job.session_id,
            "-",
        ]
    return ["codex", "exec", *config, "--sandbox", "workspace-write", "-"]


def children_of(pid: int) -> dict[int, str]:
    parents = {}
    identities = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            values = path.read_text().rsplit(")", 1)[1].split()
            child = int(path.parent.name)
            parents[child] = int(values[1])
            identities[child] = values[19]
        except (OSError, ValueError, IndexError):
            continue
    descendants = {pid}
    while True:
        expanded = descendants | {
            child for child, parent in parents.items() if parent in descendants
        }
        if expanded == descendants:
            return {child: identities[child] for child in descendants - {pid}}
        descendants = expanded


def reap() -> None:
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if not pid:
                return
        except ChildProcessError:
            return


def cleanup() -> bool:
    # A pidfd binds signals to the captured descendant, even if a PID is later reused.
    for sig, grace in ((signal.SIGTERM, 2.0), (signal.SIGKILL, 3.0)):
        deadline = time.monotonic() + grace
        while True:
            descendants = children_of(os.getpid())
            for pid, identity in descendants.items():
                try:
                    descriptor = os.pidfd_open(pid, 0)
                except ProcessLookupError:
                    continue
                except OSError:
                    return False
                try:
                    if process_start(pid) == identity:
                        signal.pidfd_send_signal(descriptor, sig)
                except ProcessLookupError:
                    pass
                except OSError:
                    return False
                finally:
                    os.close(descriptor)
            reap()
            if not children_of(os.getpid()):
                return True
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
    return False


def record_event(job: Job, event: dict) -> str | None:
    kind = event.get("type")
    if kind == "thread.started":
        try:
            job.session_id = valid_uuid(event.get("thread_id"))
        except ValueError:
            pass
        job.event("session", "Codex session recorded.")
    elif kind in {"item.started", "item.completed"}:
        item = event.get("item")
        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "agent_message" and kind == "item.completed":
                text = item.get("text")
                return final_summary(text) if isinstance(text, str) else None
            summaries = {
                "command_execution": "Codex ran a command.",
                "file_change": "Codex updated files.",
                "mcp_tool_call": "Codex used a connected tool.",
                "web_search": "Codex searched the web.",
                "todo_list": "Codex updated its work plan.",
            }
            if item_type in summaries:
                job.event("progress", summaries[item_type])
    elif kind == "turn.completed":
        job.event(
            "turn_completed", "Codex completed its turn; confirming process cleanup."
        )
    elif kind in {"turn.failed", "error"}:
        job.event(
            "error",
            "Codex reported an execution error. Inspect the computer before retrying.",
        )
    return None


def monitor(
    jobs: Jobs, job: Job, process: subprocess.Popen, interrupted=lambda: False
) -> tuple[str, str]:
    deadline = time.monotonic() + job.timeout_seconds
    summary = ""
    failed = False
    completed = False
    pending = bytearray()
    dropping = False
    selector = selectors.DefaultSelector()
    os.set_blocking(process.stdout.fileno(), False)
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        while True:
            if interrupted():
                return (
                    "interrupted",
                    "Codex supervisor was interrupted. Inspect files before retrying.",
                )
            with jobs.locked():
                saved = jobs.get(job.id)
                if saved.cancel_requested:
                    return (
                        "cancelled",
                        "Codex was cancelled. Inspect files before a new turn.",
                    )
            if time.monotonic() >= deadline:
                return (
                    "timed_out",
                    "Codex reached its timeout. Inspect files before a new turn.",
                )
            for key, _ in selector.select(0.2):
                chunk = os.read(key.fileobj.fileno(), 8192)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                for fragment in chunk.splitlines(keepends=True):
                    end = fragment.endswith(b"\n")
                    if len(pending) + len(fragment) > 65536:
                        pending.clear()
                        dropping = True
                    if not dropping:
                        pending.extend(fragment)
                    if not end:
                        continue
                    if not dropping:
                        try:
                            event = json.loads(pending)
                        except (ValueError, UnicodeError):
                            event = None
                        if isinstance(event, dict):
                            with jobs.locked():
                                job = jobs.get(job.id)
                                result = record_event(job, event)
                                if result:
                                    summary = result
                                failed |= event.get("type") in {"turn.failed", "error"}
                                completed |= event.get("type") == "turn.completed"
                                jobs.save(job)
                    pending.clear()
                    dropping = False
            code = process.poll()
            if code is not None and (not selector.get_map() or not selector.select(0)):
                if code == 0 and completed and not failed:
                    return "completed", summary or "Codex completed its turn."
                return (
                    "failed",
                    "Codex did not complete its turn. Check sign-in and inspect files before retrying.",
                )
    finally:
        selector.close()


def worker(job_id: str) -> None:
    jobs = Jobs()
    process = None
    status, message = "failed", "Codex execution could not start."
    interrupted = False

    def stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    # Failure to acquire descendant ownership must prevent any Codex process launch.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        with jobs.locked():
            job = jobs.get(job_id)
            job.status, job.message = (
                "failed",
                "Computer cannot provide coding process ownership.",
            )
            job.finished_at, job.cleanup_confirmed = now(), True
            jobs.save(job)
        (jobs.root / f"{job_id}.prompt").unlink(missing_ok=True)
        return
    with jobs.locked():
        job = jobs.get(job_id)
        if job.status not in {"starting", "stopping"}:
            return
        job.worker_pid, job.worker_start = os.getpid(), process_start(os.getpid())
        if job.cancel_requested:
            status, message = "cancelled", "Codex was cancelled before execution."
        else:
            job.status, job.message = (
                "running",
                "Codex is working in the shared computer.",
            )
            job.event("running", job.message)
        jobs.save(job)
    try:
        if not job.cancel_requested and not interrupted:
            prompt_path = jobs.root / f"{job.id}.prompt"
            with prompt_path.open("rb") as prompt:
                prompt_path.unlink()
                process = subprocess.Popen(
                    command(job),
                    cwd=job.cwd,
                    stdin=prompt,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    close_fds=True,
                    env={**os.environ, "CODEX_HOME": CODEX_HOME},
                )
            # monitor checks durable cancellation; signals are converted to the same request.
            if interrupted:
                jobs.cancel(job.id)
            status, message = monitor(jobs, job, process, lambda: interrupted)
    except Exception:
        status, message = (
            "failed",
            "Codex execution was interrupted by a supervisor error. Inspect files before retrying.",
        )
    finally:
        confirmed = cleanup()
        with jobs.locked():
            job = jobs.get(job_id)
            if interrupted:
                status, message = (
                    "interrupted",
                    "Codex supervisor was interrupted. Inspect files before retrying.",
                )
            elif job.cancel_requested:
                status, message = (
                    "cancelled",
                    "Codex was cancelled. Inspect files before a new turn.",
                )
            job.status = status if confirmed else "cleanup_required"
            job.message = (
                message
                if confirmed
                else "Codex process cleanup could not be confirmed. Restart the computer before another coding job."
            )
            job.cleanup_confirmed = confirmed
            job.finished_at = now() if confirmed else None
            job.event(
                job.status, "Coding processes stopped." if confirmed else job.message
            )
            jobs.save(job)
        (jobs.root / f"{job_id}.prompt").unlink(missing_ok=True)


def dispatch(request: dict) -> dict:
    jobs = Jobs()
    operation = request.get("operation")
    if operation == "start":
        return jobs.start(request)
    if operation == "cancel":
        return jobs.cancel(valid_uuid(request.get("job_id")))
    if operation == "status":
        after = request.get("after", 0)
        if type(after) is not int or after < 0:
            raise ValueError("Event cursor must be a nonnegative integer.")
        return jobs.status(
            valid_uuid(request["job_id"]) if request.get("job_id") else None, after
        )
    raise ValueError("Unknown coding operation.")


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "worker":
        worker(valid_uuid(sys.argv[2]))
        return
    try:
        if len(sys.argv) != 2 or len(sys.argv[1]) > 100000:
            raise ValueError("Invalid coding request.")
        request = json.loads(base64.urlsafe_b64decode(sys.argv[1]).decode())
        if not isinstance(request, dict):
            raise ValueError("Invalid coding request.")
        result = dispatch(request)
    except ValueError as error:
        result = {"error": str(error)}
    except Exception:
        result = {
            "error": "Coding supervisor is unavailable. Inspect the computer; do not automatically replay coding instructions."
        }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
