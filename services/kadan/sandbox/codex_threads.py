"""Codex sign-in, background runs and thread history on the shared Kadan computer."""

from __future__ import annotations

import base64
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
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


def alive(record: dict) -> bool:
    pid = record.get("pid")
    return isinstance(pid, int) and process_start(pid) == record.get("start")


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
    return {"pid": process.pid, "start": process_start(process.pid)}


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
    return wait_for_login()


def wait_for_login() -> dict:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        current = setup()
        if current["state"] == "signed_out":
            raise ValueError("Codex sign-in stopped before it gave a code. Try again.")
        if current["state"] != "signing_in" or "login" in current:
            return current
        time.sleep(0.25)
    terminate(login_record())
    raise ValueError("Codex didn't provide a sign-in code. Try signing in again.")


def logout() -> dict:
    terminate(login_record())
    subprocess.run(
        ["codex", "logout"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
        check=False,
        env=ENV,
    )
    return setup()


def valid_uuid(value) -> str:
    try:
        return str(UUID(value))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("A valid thread ID is required.") from None


def records() -> list[dict]:
    """Kadan-launched runs, kept after exit until Kadan acknowledges the result."""
    return [
        {**read_json(path), "id": path.stem} for path in (STATE / "runs").glob("*.json")
    ]


def runs() -> dict[str, dict]:
    """Kadan-launched runs that are still alive, by Codex thread ID."""
    return {
        record["thread_id"]: record
        for record in records()
        if record.get("thread_id") and alive(record)
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
    elif status == "failed" and not message:
        message = error
    return status, message[:4000]


def finished() -> dict:
    results = []
    for record in records():
        if alive(record) or not record.get("thread_id"):
            continue
        status, message = outcome(
            STATE / "runs" / f"{record['id']}.jsonl", bool(record.get("stopped"))
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
    for suffix in (".json", ".jsonl"):
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
        "updated_at": datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc
        ).isoformat(),
        "running": thread_id in active,
    }


def threads() -> dict:
    active = runs()
    paths = sorted(
        rollouts().items(), key=lambda item: item[1].stat().st_mtime, reverse=True
    )
    return {
        "setup": setup(),
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
    origin = request.get("origin")
    if not isinstance(origin, dict) or not all(
        isinstance(origin.get(name), str)
        for name in ("agent_key", "conversation_id", "run_id")
    ):
        raise ValueError("Codex runs need the conversation that started them.")
    root = STATE / "runs"
    existing = read_json(root / f"{request_id}.json")
    if existing.get("thread_id"):
        # A retried action returns its original run rather than starting another.
        return {
            "thread_id": existing["thread_id"],
            "cwd": existing.get("cwd", ""),
            "running": alive(existing),
        }
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
    if setup()["state"] != "signed_in":
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
    prompt_path = root / f"{request_id}.prompt"
    prompt_path.write_text(prompt)
    with prompt_path.open("rb") as stdin:
        prompt_path.unlink()
        record = {
            **launch(command, root / f"{request_id}.jsonl", cwd, stdin),
            "cwd": cwd,
            "origin": origin,
        }
    write_json(root / f"{request_id}.json", record)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        thread_id = started_thread(root / f"{request_id}.jsonl")
        if thread_id:
            write_json(root / f"{request_id}.json", {**record, "thread_id": thread_id})
            return {"thread_id": thread_id, "cwd": cwd, "running": True}
        if not alive(record):
            acknowledge(request_id)
            raise ValueError(
                "Codex exited before starting. Check that it's signed in and try again."
            )
        time.sleep(0.25)
    terminate(record)
    acknowledge(request_id)
    raise ValueError("Codex didn't start in time. Try again.")


def stop(thread_id: str) -> dict:
    record = runs().get(thread_id)
    if record is not None:
        write_json(
            STATE / "runs" / f"{record['id']}.json",
            {key: value for key, value in record.items() if key != "id"}
            | {"stopped": True},
        )
        terminate(record)
        if alive(record):
            raise ValueError("Codex is still stopping. Refresh to check again.")
    return {"thread_id": thread_id, "running": False}


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
