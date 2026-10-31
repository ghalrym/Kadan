"""Codex sign-in, background runs and thread history on the shared Kadan computer."""

from __future__ import annotations

import base64
import ctypes
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, UTC
from pathlib import Path
from uuid import UUID

CODEX_HOME = Path("/home/gem/.codex")
STATE = Path("/home/gem/.local/state/kadan/codex")
VERSION = "0.162.1"
ENV = {**os.environ, "CODEX_HOME": str(CODEX_HOME)}
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
MAX_THREADS = 50
MAX_ITEMS = 200


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


def alive(record: dict) -> bool:
    pid = record.get("pid")
    start = record.get("start")
    return (
        isinstance(pid, int)
        and pid > 1
        and isinstance(start, str)
        and record.get("incarnation") == incarnation()
        and process_start(pid) == start
    )


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def launch(command: list[str], output: Path, cwd: str, stdin=None) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=stdin if stdin is not None else subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=ENV,
        )
    return {
        "pid": process.pid,
        "start": process_start(process.pid),
        "incarnation": incarnation(),
    }


def terminate(record: dict) -> None:
    if not alive(record):
        return
    for sig, wait in ((signal.SIGTERM, 5), (signal.SIGKILL, 2)):
        try:
            os.killpg(record["pid"], sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + wait
        while alive(record) and time.monotonic() < deadline:
            time.sleep(0.1)
        if not alive(record):
            return

    raise ValueError(
        "Sign-in cleanup could not be confirmed. Restart the computer before trying again."
    )


def login_record() -> dict:
    return read_json(STATE / "login.json")


def login_prompt() -> dict:
    text = ANSI.sub("", (STATE / "login.log").read_text(errors="replace"))
    url = re.search(r"https://\S+", text)
    code = re.search(r"\b[A-Z0-9]{4,}-[A-Z0-9]{4,}\b", text)
    return {
        "url": url.group(0) if url else None,
        "code": code.group(0) if code else None,
    }


def setup() -> dict:
    binary = shutil.which("codex")
    if not binary:
        return {
            "state": "missing",
            "message": "Codex isn't installed on this computer. Ask the person who runs Kadan to update it.",
        }
    try:
        signed_in = (
            subprocess.run(
                [binary, "login", "status"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
                env=ENV,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return {
            "state": "unavailable",
            "message": "Kadan couldn't check Codex on this computer. Refresh to try again.",
        }
    login = login_record()
    if signed_in:
        terminate(login)
        return {"state": "signed_in", "message": "Signed in to Codex."}
    if alive(login):
        prompt = login_prompt()
        if prompt["url"] and prompt["code"]:
            return {
                "state": "signing_in",
                "message": "Open the sign-in page and enter the code.",
                "login": prompt,
            }
        return {"state": "signing_in", "message": "Starting sign-in…"}
    return {
        "state": "signed_out",
        "message": "Sign in with ChatGPT to let your agent use Codex.",
    }


def login() -> dict:
    current = setup()
    if current["state"] == "signed_out":
        write_json(
            STATE / "login.json",
            launch(
                ["codex", "login", "--device-auth"], STATE / "login.log", "/home/gem"
            ),
        )
    elif current["state"] != "signing_in" or "login" in current:
        return current
    return {"state": "signing_in", "message": "Starting sign-in…"}


def logout() -> dict:
    terminate(login_record())
    result = subprocess.run(
        ["codex", "logout"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
        check=False,
        env=ENV,
    )
    if result.returncode != 0:
        raise ValueError(
            "Sign-out failed. Refresh to check sign-in before trying again."
        )
    return setup()


def valid_uuid(value) -> str:
    try:
        return str(UUID(value))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("A valid thread ID is required.") from None


@contextmanager
def locked():
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "runs.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def records() -> list[dict]:
    return [
        {**read_json(path), "id": path.stem} for path in (STATE / "runs").glob("*.json")
    ]


def pending(record: dict) -> bool:
    # A dead supervisor cannot prove its descendants stopped. Only a computer
    # replacement or the supervisor's own cleanup confirmation releases ownership.
    return (
        not record.get("cleanup_confirmed")
        and record.get("incarnation") == incarnation()
    )


def runs() -> dict[str, dict]:
    return {
        (record.get("thread_id") or record["resume_thread_id"]): record
        for record in records()
        if (record.get("thread_id") or record.get("resume_thread_id"))
        and pending(record)
    }


def outcome(log: Path, stopped: bool) -> tuple[str, str]:
    status, message, error = "failed", "", "Codex exited before finishing its turn."
    try:
        with log.open() as lines:
            for line in lines:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                item = event.get("item")
                if (
                    event.get("type") == "item.completed"
                    and isinstance(item, dict)
                    and item.get("type") == "agent_message"
                    and isinstance(item.get("text"), str)
                ):
                    message = item["text"]
                elif event.get("type") == "turn.completed":
                    status = "completed"
                elif event.get("type") in {"turn.failed", "error"}:
                    status = "failed"
                    detail = event.get("error")
                    detail = detail.get("message") if isinstance(detail, dict) else None
                    error = str(detail or event.get("message") or error)
    except OSError:
        pass
    if stopped:
        status = "stopped"
    elif status == "failed":
        message = error + ("\n" + message if message else "")
    return status, message[:4000]


def finished() -> dict:
    results = []
    with locked():
        for record in records():
            if (
                pending(record)
                or record.get("acknowledged")
                or not record.get("thread_id")
            ):
                continue
            status, message = (
                record.get("status", "failed"),
                record.get("message", "Codex was interrupted by a computer restart."),
            )
            if not record.get("cleanup_confirmed"):
                status, message = (
                    "failed",
                    "Codex was interrupted by a computer restart. Inspect the project before continuing.",
                )
            results.append(
                {
                    "request_id": record["id"],
                    "thread_id": record["thread_id"],
                    "cwd": record.get("cwd", ""),
                    "origin": record.get("origin"),
                    "status": status,
                    "message": message,
                }
            )
    return {"runs": results}


def acknowledge(request_id: str) -> dict:
    with locked():
        path = STATE / "runs" / f"{request_id}.json"
        record = read_json(path)
        if record:
            if pending(record):
                raise ValueError(
                    "Codex is still working; its result cannot be acknowledged."
                )
            # Retain the action receipt: a runner may replay the original start
            # even after its completion event has been accepted by Tsuchi.
            write_json(path, {**record, "acknowledged": True})
        for suffix in (".jsonl", ".worker.log"):
            (STATE / "runs" / f"{request_id}{suffix}").unlink(missing_ok=True)
    return {"request_id": request_id}


def started_thread(log: Path) -> str | None:
    try:
        with log.open() as lines:
            for line in lines:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict) and event.get("type") == "thread.started":
                    return valid_uuid(event.get("thread_id"))
    except (OSError, ValueError):
        pass
    return None


def rollouts() -> dict[str, Path]:
    found = {}
    for path in (CODEX_HOME / "sessions").glob("*/*/*/rollout-*.jsonl"):
        match = re.search(
            r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.jsonl$",
            path.name,
        )
        if match:
            found[match.group(1)] = path
    return found


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


def command_text(payload: dict) -> str | None:
    kind = payload.get("type")
    if kind == "custom_tool_call" and payload.get("name") == "apply_patch":
        return "Edited files"
    if kind == "local_shell_call":
        command = (payload.get("action") or {}).get("command")
    elif kind == "function_call":
        try:
            arguments = json.loads(payload.get("arguments") or "{}")
        except ValueError:
            arguments = {}
        command = arguments.get("command") or arguments.get("cmd")
        if command is None:
            return f"Used {payload.get('name', 'a tool')}"
    else:
        return None
    if isinstance(command, list):
        command = (
            command[-1]
            if command[:2] == ["bash", "-lc"]
            else " ".join(map(str, command))
        )
    return f"Ran {str(command)[:300]}" if command else None


def parse(path: Path) -> dict:
    thread = {"cwd": "", "created_at": None, "title": "", "items": []}
    items = thread["items"]
    try:
        with path.open() as lines:
            for line in lines:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(entry, dict) or not isinstance(
                    entry.get("payload"), dict
                ):
                    continue
                payload = entry["payload"]
                if entry.get("type") == "session_meta":
                    thread["cwd"] = str(payload.get("cwd") or "")
                    thread["created_at"] = payload.get("timestamp") or entry.get(
                        "timestamp"
                    )
                elif entry.get("type") == "event_msg" and payload.get("type") in {
                    "user_message",
                    "agent_message",
                }:
                    text = text_of(payload.get("message")).strip()
                    if text:
                        role = (
                            "user" if payload["type"] == "user_message" else "assistant"
                        )
                        items.append({"kind": role, "text": text[:20000]})
                        if role == "user" and not thread["title"]:
                            thread["title"] = " ".join(text.split())[:120]
                elif entry.get("type") == "response_item":
                    text = command_text(payload)
                    if text:
                        items.append({"kind": "command", "text": text})
    except OSError:
        pass
    return thread


def summary(thread_id: str, path: Path, active: dict[str, dict]) -> dict:
    thread = parse(path)
    return {
        "id": thread_id,
        "title": thread["title"] or "Untitled thread",
        "cwd": thread["cwd"],
        "created_at": thread["created_at"],
        "updated_at": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
        "running": thread_id in active,
    }


def threads() -> dict:
    active = runs()
    paths = sorted(
        rollouts().items(), key=lambda item: item[1].stat().st_mtime, reverse=True
    )
    return {
        "threads": [
            summary(thread_id, path, active) for thread_id, path in paths[:MAX_THREADS]
        ],
    }


def thread(thread_id: str) -> dict:
    path = rollouts().get(thread_id)
    if path is None:
        raise ValueError("That Codex thread isn't on this computer.")
    items = parse(path)["items"]
    return {
        **summary(thread_id, path, runs()),
        "items": items[-MAX_ITEMS:],
        "items_truncated": len(items) > MAX_ITEMS,
    }


def start(request: dict) -> dict:
    request_id = valid_uuid(request.get("request_id"))
    with locked():
        reserve(request, request_id)
    deadline = time.monotonic() + 20
    path = STATE / "runs" / f"{request_id}.json"
    while time.monotonic() < deadline:
        record = read_json(path)
        if record.get("thread_id"):
            return {
                "thread_id": record["thread_id"],
                "cwd": record["cwd"],
                "running": pending(record),
            }
        if record.get("cleanup_confirmed"):
            raise ValueError(
                record.get("message")
                or "Codex could not start. Inspect the project before another request."
            )
        time.sleep(0.1)
    raise ValueError(
        "Codex start is unconfirmed. Check Coding before submitting another task."
    )


def reserve(request: dict, request_id: str) -> None:
    origin = request.get("origin")
    if not isinstance(origin, dict) or not all(
        isinstance(origin.get(name), str)
        for name in ("agent_key", "conversation_id", "run_id")
    ):
        raise ValueError("Codex runs need the conversation that started them.")
    root = STATE / "runs"
    if read_json(root / f"{request_id}.json"):
        return
    if any(
        pending(record) and (not alive(record) or not record.get("thread_id"))
        for record in records()
    ):
        raise ValueError(
            "Coding cleanup is unconfirmed. Restart the computer before another task."
        )
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Give Codex a task.")
    resume = valid_uuid(request["thread_id"]) if request.get("thread_id") else None
    if resume:
        if resume in runs():
            raise ValueError(
                "That thread is already running. Wait for it or stop it first."
            )
        path = rollouts().get(resume)
        if path is None:
            raise ValueError("That Codex thread isn't on this computer.")
        cwd = parse(path)["cwd"]
    else:
        cwd = request.get("cwd")
    if not isinstance(cwd, str) or not os.path.isabs(cwd):
        raise ValueError("Use an absolute project folder on this computer.")
    with (STATE / "login.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        signed_in = setup()["state"] == "signed_in"
    if not signed_in:
        raise ValueError(
            "Codex isn't signed in. Ask the user to sign in from the Coding tab."
        )
    if not resume:
        # A new project can start in a folder that doesn't exist yet.
        try:
            os.makedirs(cwd, exist_ok=True)
        except OSError:
            raise ValueError("Kadan couldn't create that project folder.") from None
    if not os.path.isdir(cwd):
        raise ValueError("That thread's project folder no longer exists.")
    options = [
        "--json",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "-c",
        'model_reasoning_effort="high"',
        "-c",
        'approval_policy="never"',
        "-c",
        'sandbox_mode="workspace-write"',
        "-m",
        "gpt-6.1-sol",
    ]
    command = (
        ["codex", "exec", "resume", *options, resume, "-"]
        if resume
        else ["codex", "exec", *options, "-"]
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{request_id}.json"
    record = {
        "cwd": cwd,
        "origin": origin,
        "incarnation": incarnation(),
        "command": command,
        "resume_thread_id": resume,
    }
    (root / f"{request_id}.prompt").write_text(prompt)
    # Reserve before spawning. A worker writes its identity under the same lock
    # before it can launch Codex, including if the requesting Bash session dies.
    write_json(path, record)
    try:
        worker = launch(
            [sys.executable, str(Path(__file__).resolve()), "--worker", request_id],
            root / f"{request_id}.worker.log",
            cwd,
        )
        write_json(path, {**record, **worker})
    except OSError:
        write_json(
            path,
            {
                **record,
                "cleanup_confirmed": True,
                "status": "failed",
                "message": "Codex supervisor could not start.",
            },
        )
        (root / f"{request_id}.prompt").unlink(missing_ok=True)
        raise ValueError("Codex supervisor could not start.") from None


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


def worker(request_id: str) -> None:
    path = STATE / "runs" / f"{request_id}.json"
    log = STATE / "runs" / f"{request_id}.jsonl"
    interrupted = False

    def interrupt(_signal, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    process = None
    failed = False
    owned = ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0
    with locked():
        record = read_json(path)
        if record.get("cleanup_confirmed") or (
            alive(record) and record.get("pid") != os.getpid()
        ):
            return
        record.update(
            pid=os.getpid(),
            start=process_start(os.getpid()),
            incarnation=incarnation(),
        )
        write_json(path, record)
    try:
        if interrupted or record.get("stopped"):
            return
        if not owned:
            raise ValueError("Computer cannot provide coding process ownership.")
        with (
            (STATE / "runs" / f"{request_id}.prompt").open("rb") as prompt,
            log.open("wb") as output,
        ):
            process = subprocess.Popen(
                record["command"],
                cwd=record["cwd"],
                stdin=prompt,
                stdout=output,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
                env=ENV,
            )
        (STATE / "runs" / f"{request_id}.prompt").unlink(missing_ok=True)
        deadline = time.monotonic() + 20
        while process.poll() is None:
            with locked():
                record = read_json(path)
                thread_id = record.get("thread_id") or started_thread(log)
                if thread_id and not record.get("thread_id"):
                    record["thread_id"] = thread_id
                    write_json(path, record)
            if (
                interrupted
                or record.get("stopped")
                or (not thread_id and time.monotonic() >= deadline)
            ):
                break
            time.sleep(0.1)
    except Exception:
        failed = True
    finally:
        exit_code = process.poll() if process is not None else None
        confirmed = cleanup() if owned else process is None
        with locked():
            record = read_json(path)
            thread_id = record.get("thread_id") or started_thread(log)
            status, message = outcome(log, bool(record.get("stopped")))
            if not record.get("stopped") and (
                failed or interrupted or process is None or exit_code != 0
            ):
                if status == "completed" or not message:
                    message = "Codex execution was interrupted or failed. Inspect the project before another task."
                status = "failed"
            write_json(
                path,
                {
                    **record,
                    "thread_id": thread_id,
                    "status": status,
                    "message": message,
                    "cleanup_confirmed": confirmed,
                },
            )
        (STATE / "runs" / f"{request_id}.prompt").unlink(missing_ok=True)


def stop(thread_id: str) -> dict:
    with locked():
        record = runs().get(thread_id)
        if record is None:
            return {"thread_id": thread_id, "running": False}
        path = STATE / "runs" / f"{record['id']}.json"
        write_json(
            path,
            {key: value for key, value in record.items() if key != "id"}
            | {"stopped": True},
        )
    deadline = time.monotonic() + 7
    while time.monotonic() < deadline:
        if not pending(read_json(path)):
            return {"thread_id": thread_id, "running": False}
        time.sleep(0.1)
    raise ValueError(
        "Codex cleanup is unconfirmed. Refresh Coding; restart the computer if it remains working."
    )


def dispatch(request: dict) -> dict:
    operation = request.get("operation")
    if operation == "setup":
        return setup()
    if operation == "login":
        return login()
    if operation == "logout":
        return logout()
    if operation == "threads":
        return threads()
    if operation == "thread":
        return thread(valid_uuid(request.get("thread_id")))
    if operation == "start":
        return start(request)
    if operation == "stop":
        return stop(valid_uuid(request.get("thread_id")))
    if operation == "finished":
        return finished()
    if operation == "acknowledge":
        return acknowledge(valid_uuid(request.get("request_id")))
    raise ValueError("Unknown Codex operation.")


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(valid_uuid(sys.argv[2]))
        return
    try:
        # Codex refuses to run when CODEX_HOME is missing, as on a fresh home volume.
        CODEX_HOME.mkdir(parents=True, exist_ok=True)
        if len(sys.argv) != 2 or len(sys.argv[1]) > 100000:
            raise ValueError("Invalid Codex request.")
        request = json.loads(base64.urlsafe_b64decode(sys.argv[1]).decode())
        if not isinstance(request, dict):
            raise ValueError("Invalid Codex request.")
        if request.get("operation") in {"setup", "login", "logout"}:
            STATE.mkdir(parents=True, exist_ok=True)
            with (STATE / "login.lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                result = dispatch(request)
        else:
            result = dispatch(request)
    except ValueError as error:
        result = {"error": str(error)}
    except Exception:
        result = {
            "error": "Codex is unavailable on this computer. Refresh to try again."
        }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
