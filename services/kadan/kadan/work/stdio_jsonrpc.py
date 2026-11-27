"""Shared newline-delimited stdio JSON-RPC transport for coding harness CLIs."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from .contract import HarnessError

OUTPUT_LINE_LIMIT = 16 * 1024 * 1024
ERROR_TAIL_CHARS = 2000


class StdioProcess:
    """Thin wrapper around an asyncio subprocess with line-oriented JSON I/O."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        *,
        stderr_task: asyncio.Task[bytes] | None = None,
    ):
        self.process = process
        self._stderr_task = stderr_task

    @classmethod
    async def spawn(
        cls,
        command: list[str],
        *,
        cwd: str,
        env: dict[str, str] | None = None,
        not_found_message: str = "The coding harness CLI is not installed on this server.",
    ) -> StdioProcess:
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=OUTPUT_LINE_LIMIT,
            )
        except FileNotFoundError as exc:
            raise HarnessError(not_found_message) from exc
        stderr_task = (
            asyncio.create_task(proc.stderr.read()) if proc.stderr is not None else None
        )
        return cls(proc, stderr_task=stderr_task)

    async def write_json(self, frame: dict[str, Any]) -> None:
        if self.process.stdin is None:
            raise HarnessError("Harness input stream is closed.")
        data = (json.dumps(frame, ensure_ascii=False) + "\n").encode()
        self.process.stdin.write(data)
        await self.process.stdin.drain()

    async def read_json(self) -> dict[str, Any] | None:
        """Read one JSON object line from stdout; returns None on clean EOF."""
        if self.process.stdout is None:
            return None
        while True:
            line = await self.process.stdout.readline()
            if not line:
                return None
            text = line.decode(errors="replace").strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except ValueError:
                # Ignore stray non-JSON diagnostic lines emitted by some CLIs.
                continue
            if isinstance(parsed, dict):
                return parsed

    async def error_summary(self) -> str:
        if self.process.returncode is None:
            with suppress(TimeoutError):
                await asyncio.wait_for(self.process.wait(), timeout=2.0)
        stderr = ""
        if self._stderr_task is not None:
            with suppress(asyncio.CancelledError, Exception):
                if self._stderr_task.done():
                    stderr = self._stderr_task.result().decode(errors="replace").strip()
                else:
                    raw = await asyncio.wait_for(self._stderr_task, timeout=1.0)
                    stderr = raw.decode(errors="replace").strip()
        if stderr:
            return stderr[-ERROR_TAIL_CHARS:]
        if self.process.returncode is not None:
            return f"The coding harness exited with status {self.process.returncode}."
        return "The coding harness connection closed unexpectedly."

    async def close(self) -> None:
        if self.process.stdin is not None:
            with suppress(Exception):
                self.process.stdin.close()
        if self.process.returncode is None:
            with suppress(ProcessLookupError):
                self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=3.0)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    self.process.kill()
                await self.process.wait()
        if self._stderr_task is not None:
            if not self._stderr_task.done():
                self._stderr_task.cancel()
            await asyncio.gather(self._stderr_task, return_exceptions=True)


ProcessFactory = Callable[
    [list[str], str, dict[str, str] | None], Awaitable[StdioProcess]
]


class JsonRpcStdioClient:
    """Bidirectional JSON-RPC 2.0 client over newline-delimited stdio."""

    def __init__(self, transport: StdioProcess, *, include_jsonrpc: bool = True):
        self.transport = transport
        self.include_jsonrpc = include_jsonrpc
        self._seq = 0
        self._pending: dict[int | str, asyncio.Future[Any]] = {}
        self.incoming: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._reader_task: asyncio.Task[None] | None = None
        self._closed = False

    def start(self) -> None:
        if self._reader_task is None:
            self._reader_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        try:
            while True:
                frame = await self.transport.read_json()
                if frame is None:
                    break
                if "method" not in frame and "id" in frame:
                    fut = self._pending.pop(frame["id"], None)
                    if fut is not None and not fut.done():
                        if "error" in frame and frame["error"] is not None:
                            err = frame["error"]
                            msg = (
                                err.get("message")
                                if isinstance(err, dict) and err.get("message")
                                else str(err)
                            )
                            fut.set_exception(HarnessError(str(msg)[:ERROR_TAIL_CHARS]))
                        else:
                            fut.set_result(frame.get("result"))
                    continue
                await self.incoming.put(frame)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(HarnessError(str(exc)))
            self._pending.clear()
        finally:
            if self._pending:
                detail = await self.transport.error_summary()
                for fut in list(self._pending.values()):
                    if not fut.done():
                        fut.set_exception(HarnessError(detail))
                self._pending.clear()
            await self.incoming.put(None)

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = 60.0,
    ) -> Any:
        self.start()
        self._seq += 1
        req_id = self._seq
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        self._pending[req_id] = fut
        frame: dict[str, Any] = {"id": req_id, "method": method}
        if self.include_jsonrpc:
            frame["jsonrpc"] = "2.0"
        if params is not None:
            frame["params"] = params
        await self.transport.write_json(frame)
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except TimeoutError as exc:
            self._pending.pop(req_id, None)
            raise HarnessError(f"Timed out waiting for '{method}' response.") from exc

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        frame: dict[str, Any] = {"method": method}
        if self.include_jsonrpc:
            frame["jsonrpc"] = "2.0"
        if params is not None:
            frame["params"] = params
        await self.transport.write_json(frame)

    async def respond(
        self,
        request_id: int | str,
        *,
        result: Any = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        frame: dict[str, Any] = {"id": request_id}
        if self.include_jsonrpc:
            frame["jsonrpc"] = "2.0"
        if error is not None:
            frame["error"] = error
        else:
            frame["result"] = result if result is not None else {}
        await self.transport.write_json(frame)

    async def next_message(self) -> dict[str, Any] | None:
        self.start()
        return await self.incoming.get()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            await asyncio.gather(self._reader_task, return_exceptions=True)
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        await self.transport.close()
