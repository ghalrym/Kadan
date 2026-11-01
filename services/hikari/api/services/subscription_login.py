"""Web sign-in for subscriptions, driving each provider CLI's own login flow.

Settings shows only the sign-in link and, for Codex, the one-time code. Claude's
login hands the user a code on its web page, which Settings sends back here.
"""

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

from api.pydantic_models.subscriptions import SubscriptionLogin, SubscriptionProvider
from api.repositories import subscriptions

LOGIN_SECONDS = 15 * 60
START_SECONDS = 30
OUTPUT_BYTES = 1024 * 1024
# Claude's sign-in is an interactive terminal UI; a wide terminal keeps its link on one line.
TERMINAL_SIZE = struct.pack("HHHH", 50, 2000, 0, 0)

ANSI = re.compile(
    r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-Z\\-_]"
)
CLAUDE_URL = re.compile(r"https://\S+/oauth/authorize\?\S+")
CLAUDE_TOKEN = re.compile(r"sk-ant-oat\d+-[A-Za-z0-9_-]+")
CLAUDE_ERROR = re.compile(r"OAuth error")
CODEX_URL = re.compile(r"https://auth\.openai\.com/\S+")
CODEX_CODE = re.compile(r"one-time code.*?\b([A-Z0-9]{4}-[A-Z0-9]{4,6})\b", re.DOTALL)


class LoginFailure(Exception):
    pass


def clean(raw: bytes) -> str:
    return ANSI.sub(" ", raw.decode(errors="replace"))


class LoginRun:
    """One provider's sign-in; it saves the subscription itself when the user finishes."""

    def __init__(self, provider: SubscriptionProvider):
        self.provider = provider
        self.state = SubscriptionLogin(
            provider=provider, status="starting", needs_code=provider == "claude"
        )
        self.raw = b""
        self.closed = False
        self.changed = asyncio.Event()
        self.home = Path(tempfile.mkdtemp(prefix=f"hikari-login-{provider}-"))
        self.process: asyncio.subprocess.Process | None = None
        self.terminal: int | None = None
        self.tasks: list[asyncio.Task] = []

    def text(self) -> str:
        return clean(self.raw)

    def _received(self, data: bytes) -> None:
        if data:
            self.raw = (self.raw + data)[-OUTPUT_BYTES:]
        else:
            self.closed = True
        self.changed.set()

    def _read_terminal(self) -> None:
        try:
            data = os.read(self.terminal, 65536)
        except OSError:
            data = b""
        if not data:
            asyncio.get_running_loop().remove_reader(self.terminal)
        self._received(data)

    async def _read_pipe(self) -> None:
        while data := await self.process.stdout.read(65536):
            self._received(data)
        self._received(b"")

    async def _wait_for(self, find):
        while True:
            found = find(self.text())
            if found:
                return found
            if self.closed:
                raise LoginFailure(self._failure())
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

    async def start(self) -> SubscriptionLogin:
        env = {**os.environ, "HOME": str(self.home), "BROWSER": "true", "NO_COLOR": "1"}
        try:
            if self.provider == "claude":
                env.update(COLUMNS="2000", LINES="50")
                self.terminal, child = pty.openpty()
                fcntl.ioctl(child, termios.TIOCSWINSZ, TERMINAL_SIZE)
                try:
                    self.process = await asyncio.create_subprocess_exec(
                        "claude",
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
                    "codex",
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
                    self.state.url = (await self._wait_for(CLAUDE_URL.search)).group(0)
                else:
                    self.state.url = (await self._wait_for(CODEX_URL.search)).group(0)
                    self.state.user_code = (
                        await self._wait_for(CODEX_CODE.search)
                    ).group(1)
        except FileNotFoundError:
            await self._fail("Sign-in is not available in this Hikari installation.")
            return self.state
        except (TimeoutError, LoginFailure) as error:
            await self._fail(
                str(error) or "The sign-in page did not open. Please try again."
            )
            return self.state
        self.state.status = "waiting"
        self.tasks.append(asyncio.create_task(self._finish()))
        return self.state

    async def submit_code(self, code: str) -> SubscriptionLogin:
        if self.provider != "claude" or self.state.status != "waiting":
            raise LoginFailure("This sign-in is not waiting for a code.")
        os.write(self.terminal, code.strip().encode() + b"\r")
        self.state.status = "verifying"
        return self.state

    @staticmethod
    def _claude_token(text: str):
        if CLAUDE_ERROR.search(text):
            raise LoginFailure("Claude did not accept that code. Please start again.")
        return CLAUDE_TOKEN.search(text)

    async def _finish(self) -> None:
        try:
            async with asyncio.timeout(LOGIN_SECONDS):
                if self.provider == "claude":
                    credential = (await self._wait_for(self._claude_token)).group(0)
                else:
                    if await self.process.wait() != 0:
                        raise LoginFailure(self._failure())
                    credential = (self.home / "auth.json").read_text()
            await asyncio.to_thread(subscriptions.store, self.provider, credential)
            self.state.status = "complete"
        except TimeoutError:
            self.state.status, self.state.error = (
                "failed",
                "Sign-in timed out. Please start again.",
            )
        except (LoginFailure, OSError, subscriptions.SubscriptionError) as error:
            self.state.status, self.state.error = "failed", str(error)
        finally:
            await self.close()

    async def _fail(self, message: str) -> None:
        self.state.status, self.state.error = "failed", message
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
            self.state.status, self.state.error = "failed", "Sign-in was cancelled."
        shutil.rmtree(self.home, ignore_errors=True)


class SubscriptionLogins:
    """At most one sign-in per provider; starting again replaces the earlier one."""

    def __init__(self):
        self.runs: dict[SubscriptionProvider, LoginRun] = {}

    async def start(self, provider: SubscriptionProvider) -> SubscriptionLogin:
        await self.cancel(provider)
        run = self.runs[provider] = LoginRun(provider)
        return await run.start()

    def status(self, provider: SubscriptionProvider) -> SubscriptionLogin | None:
        run = self.runs.get(provider)
        return run.state if run else None

    async def submit_code(
        self, provider: SubscriptionProvider, code: str
    ) -> SubscriptionLogin:
        run = self.runs.get(provider)
        if run is None:
            raise LoginFailure("Start signing in first.")
        return await run.submit_code(code)

    async def cancel(self, provider: SubscriptionProvider) -> None:
        run = self.runs.pop(provider, None)
        if run is not None:
            await run.close()

    async def close(self) -> None:
        for provider in tuple(self.runs):
            await self.cancel(provider)


logins = SubscriptionLogins()
