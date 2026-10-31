"""Manage Codex device sign-in on the shared Kadan computer."""

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
from pathlib import Path

CODEX_HOME = Path("/home/gem/.codex")
STATE = Path("/home/gem/.local/state/kadan/codex")
ENV = {**os.environ, "CODEX_HOME": str(CODEX_HOME)}
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


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


def main() -> None:
    try:
        # Codex refuses to run when CODEX_HOME is missing, as on a fresh home volume.
        CODEX_HOME.mkdir(parents=True, exist_ok=True)
        if len(sys.argv) != 2 or len(sys.argv[1]) > 100000:
            raise ValueError("Invalid Codex request.")
        request = json.loads(base64.urlsafe_b64decode(sys.argv[1]).decode())
        if not isinstance(request, dict):
            raise ValueError("Invalid Codex request.")
        operation = request.get("operation")
        if operation not in {"setup", "login", "logout"}:
            raise ValueError("Unknown Codex operation.")
        STATE.mkdir(parents=True, exist_ok=True)
        with (STATE / "login.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = {"setup": setup, "login": login, "logout": logout}[operation]()
    except ValueError as error:
        result = {"error": str(error)}
    except Exception:
        result = {
            "error": "Codex is unavailable on this computer. Refresh to try again."
        }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
