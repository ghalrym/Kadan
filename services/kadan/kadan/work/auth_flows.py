"""UI-driven browser sign-in flows for Codex (`codex login --device-auth`) and Claude Code (`claude setup-token`).

Mirrors Hikari's web sign-in flow so users never touch terminals or config files,
and stores credentials durably in Tsuchi's `/api/work/harnesses/{harness}` table.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import pty
import re
import shutil
import struct
import tempfile
import termios
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ..api.config import ServiceError
from .contract import HarnessAuthStatus, HarnessError

if TYPE_CHECKING:
    from ..api.services import LocalServices

LOGIN_SECONDS = 15 * 60
START_SECONDS = 30
OUTPUT_BYTES = 1024 * 1024
TERMINAL_SIZE = struct.pack("HHHH", 50, 2000, 0, 0)

ANSI = re.compile(
    r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-Z\\-_]"
)
CLAUDE_URL = re.compile(r"https://\S+/oauth/authorize\?\S+")
CLAUDE_TOKEN = re.compile(r"sk-ant-oat\d+-[A-Za-z0-9_-]+")
CLAUDE_ERROR = re.compile(r"OAuth error")
CODEX_URL = re.compile(r"https://auth\.openai\.com/\S+")
CODEX_CODE = re.compile(r"one-time code.*?\b([A-Z0-9]{4}-[A-Z0-9]{4,6})\b", re.DOTALL)


def strip_ansi(raw: bytes) -> str:
    return ANSI.sub(" ", raw.decode(errors="replace"))


class CliLoginRun:
    """Drives one interactive CLI sign-in and persists the credential to Tsuchi."""

    def __init__(
        self,
        provider: Literal["codex", "claude"],
        binary: str,
        services: LocalServices,
    ):
        self.provider = provider
        self.binary = binary
        self.services = services
        self.state = HarnessAuthStatus(
            status="starting",
            login_method="oauth_code" if provider == "claude" else "device_code",
            needs_code=provider == "claude",
            message="Starting sign-in…",
        )
        self.raw = b""
        self.closed = False
        self.changed = asyncio.Event()
        self.home = Path(tempfile.mkdtemp(prefix=f"kadan-work-login-{provider}-"))
        self.process: asyncio.subprocess.Process | None = None
        self.terminal: int | None = None
        self.tasks: list[asyncio.Task[None]] = []

    def text(self) -> str:
        return strip_ansi(self.raw)

    def _received(self, data: bytes) -> None:
        if data:
            self.raw = (self.raw + data)[-OUTPUT_BYTES:]
        else:
            self.closed = True
        self.changed.set()

    def _read_terminal(self) -> None:
        assert self.terminal is not None
        try:
            data = os.read(self.terminal, 65536)
        except OSError:
            data = b""
        if not data:
            with suppress(ValueError, OSError):
                asyncio.get_running_loop().remove_reader(self.terminal)
        self._received(data)

    async def _read_pipe(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        while data := await self.process.stdout.read(65536):
            self._received(data)
        self._received(b"")

    async def _wait_for(self, find):
        while True:
            found = find(self.text())
            if found:
                return found
            if self.closed:
                raise HarnessError(self._failure())
            self.changed.clear()
            await self.changed.wait()

    def _failure(self) -> str:
        lines = [
            line.strip() for line in self.text().splitlines() if len(line.strip()) > 3
        ]
        detail = lines[-1][:300] if lines else ""
        return (
            "Sign-in did not finish."
            + (f" {detail}" if detail else "")
            + " Please try again."
        )

    async def start(self) -> HarnessAuthStatus:
        env = {**os.environ, "HOME": str(self.home), "BROWSER": "true", "NO_COLOR": "1"}
        try:
            if self.provider == "claude":
                env.update(COLUMNS="2000", LINES="50")
                self.terminal, child = pty.openpty()
                fcntl.ioctl(child, termios.TIOCSWINSZ, TERMINAL_SIZE)
                try:
                    self.process = await asyncio.create_subprocess_exec(
                        self.binary,
                        "setup-token",
                        stdin=child,
                        stdout=child,
                        stderr=child,
                        env=env,
                        cwd=self.home,
                        start_new_session=True,
                    )
                finally:
                    os.close(child)
                os.set_blocking(self.terminal, False)
                asyncio.get_running_loop().add_reader(
                    self.terminal, self._read_terminal
                )
            else:
                env["CODEX_HOME"] = str(self.home)
                self.process = await asyncio.create_subprocess_exec(
                    self.binary,
                    "login",
                    "--device-auth",
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=env,
                    cwd=self.home,
                )
                self.tasks.append(asyncio.create_task(self._read_pipe()))

            async with asyncio.timeout(START_SECONDS):
                if self.provider == "claude":
                    self.state.authorize_url = (
                        await self._wait_for(CLAUDE_URL.search)
                    ).group(0)
                    self.state.message = "Open the sign-in link in your browser, then paste the code below."
                else:
                    self.state.authorize_url = (
                        await self._wait_for(CODEX_URL.search)
                    ).group(0)
                    self.state.user_code = (
                        await self._wait_for(CODEX_CODE.search)
                    ).group(1)
                    self.state.message = (
                        "Open the sign-in link and enter the one-time code to finish."
                    )
        except FileNotFoundError:
            await self._fail(
                f"{self.provider.title()} is not installed on this server."
            )
            return self.state
        except (TimeoutError, HarnessError) as error:
            await self._fail(
                str(error) or "The sign-in page did not open. Please try again."
            )
            return self.state

        self.state.status = "waiting"
        self.tasks.append(asyncio.create_task(self._finish()))
        return self.state

    async def submit_code(self, code: str) -> HarnessAuthStatus:
        if self.provider != "claude" or self.state.status != "waiting":
            raise HarnessError("This sign-in is not waiting for a code.")
        if self.terminal is None:
            raise HarnessError("Sign-in session is no longer active.")
        os.write(self.terminal, code.strip().encode() + b"\r")
        self.state.status = "verifying"
        self.state.message = "Verifying your sign-in code…"
        return self.state

    @staticmethod
    def _claude_token(text: str):
        if CLAUDE_ERROR.search(text):
            raise HarnessError("Claude did not accept that code. Please start again.")
        return CLAUDE_TOKEN.search(text)

    async def _finish(self) -> None:
        try:
            async with asyncio.timeout(LOGIN_SECONDS):
                if self.provider == "claude":
                    credential = (await self._wait_for(self._claude_token)).group(0)
                    auth_mode = "oauth_token"
                    label = "Claude subscription"
                else:
                    assert self.process is not None
                    if await self.process.wait() != 0:
                        raise HarnessError(self._failure())
                    credential = (self.home / "auth.json").read_text()
                    auth_mode = "device_auth"
                    label = "ChatGPT Codex"
            await self.services.json_request(
                "Tsuchi",
                "PUT",
                f"/api/work/harnesses/{self.provider}",
                json={
                    "credential": credential,
                    "auth_mode": auth_mode,
                    "account_label": label,
                    "metadata": {},
                },
            )
            self.state.status = "ready"
            self.state.account_label = label
            self.state.authorize_url = None
            self.state.user_code = None
            self.state.message = f"Signed in to {label}."
        except TimeoutError:
            self.state.status = "error"
            self.state.message = "Sign-in timed out. Please start again."
        except (HarnessError, OSError, ServiceError) as error:
            self.state.status = "error"
            self.state.message = str(error)
        finally:
            await self.close()

    async def _fail(self, message: str) -> None:
        self.state.status = "error"
        self.state.message = message
        await self.close()

    async def close(self) -> None:
        if self.terminal is not None:
            with suppress(ValueError, OSError):
                asyncio.get_running_loop().remove_reader(self.terminal)
            with suppress(OSError):
                os.close(self.terminal)
            self.terminal = None
        if self.process is not None and self.process.returncode is None:
            with suppress(ProcessLookupError):
                self.process.kill()
            await self.process.wait()
        current = asyncio.current_task()
        for task in self.tasks:
            if task is not current and not task.done():
                task.cancel()
        if self.state.status in ("starting", "waiting", "verifying"):
            self.state.status = "error"
            self.state.message = "Sign-in was cancelled."
        shutil.rmtree(self.home, ignore_errors=True)


class CliLoginManager:
    """Tracks at most one in-flight browser sign-in per harness."""

    def __init__(self) -> None:
        self._runs: dict[str, CliLoginRun] = {}

    async def start(
        self,
        provider: Literal["codex", "claude"],
        binary: str,
        services: LocalServices,
    ) -> HarnessAuthStatus:
        await self.cancel(provider)
        run = CliLoginRun(provider, binary, services)
        self._runs[provider] = run
        return await run.start()

    def status(self, provider: str) -> HarnessAuthStatus | None:
        run = self._runs.get(provider)
        return run.state if run is not None else None

    async def submit_code(self, provider: str, code: str) -> HarnessAuthStatus:
        run = self._runs.get(provider)
        if run is None:
            raise HarnessError("Start signing in first.")
        return await run.submit_code(code)

    async def cancel(self, provider: str) -> None:
        run = self._runs.pop(provider, None)
        if run is not None:
            await run.close()


cli_logins = CliLoginManager()
