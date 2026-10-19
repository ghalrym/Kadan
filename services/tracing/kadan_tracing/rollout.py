"""Local rollout admission and observation; control stays on the Compose host."""

import asyncio
import fcntl
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.request import ProxyHandler, build_opener


class Admission:
    def __init__(self, service=None, fence=None):
        self.service = service or os.getenv("KADAN_ROLLOUT_SERVICE", "")
        self.fence = Path(
            fence or os.getenv("KADAN_ROLLOUT_FENCE", "/run/kadan/rollout/fence.json")
        )
        self.active = 0
        self.check_ready = lambda: True
        self.extra_active = lambda: 0
        self.extra_status = dict
        self.prove_idle = None

    @property
    def fenced(self):
        if not self.service:
            return False
        try:
            value = json.loads(self.fence.read_text())
            closed = value["closed"]
            if not isinstance(closed, list) or not all(
                isinstance(item, str) for item in closed
            ):
                return True
            return self.service in closed
        except (OSError, ValueError, KeyError, TypeError):
            # Enrollment requires a valid host-owned gate, including after restart.
            return True

    def enter(self):
        # No await between checking the persistent fence and recording admission.
        if self.fenced:
            return False
        self.active += 1
        return True

    def leave(self):
        self.active -= 1

    def status(self):
        return {
            "fenced": self.fenced,
            "active": self.active + self.extra_active(),
            "ready": bool(self.service) and self.check_ready(),
            **self.extra_status(),
        }

    @asynccontextmanager
    async def observe(self):
        if not self.service:
            yield
            return
        # Local nonblocking ownership check runs once during startup.
        with Path("/tmp/kadan-rollout.lock").open("a") as lock:  # noqa: ASYNC230
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise RuntimeError(
                    "Rollout admission requires one application process per container"
                ) from None
            socket = Path("/tmp/kadan-rollout.sock")
            socket.unlink(missing_ok=True)

            async def respond(reader, writer):
                try:
                    command = (await asyncio.wait_for(reader.readline(), 3)).strip()
                    if command == b"status":
                        result = self.status()
                    elif command == b"barrier" and self.prove_idle is not None:
                        try:
                            result = await asyncio.wait_for(self.prove_idle(), 25)
                        except Exception:  # noqa: BLE001 - fixed local command fails closed
                            result = {
                                "native_idle": False,
                                "error": "Native idle barrier failed.",
                            }
                    else:
                        result = {"error": "Unsupported rollout observation command."}
                    writer.write(json.dumps(result).encode() + b"\n")
                    await writer.drain()
                finally:
                    writer.close()
                    await writer.wait_closed()

            server = await asyncio.start_unix_server(respond, path=socket)
            socket.chmod(0o600)
            try:
                yield
            finally:
                server.close()
                await server.wait_closed()
                socket.unlink(missing_ok=True)


admission = Admission()


class RolloutAdmission:
    """Count the entire ASGI response, including streaming and background work."""

    def __init__(self, app, gate=None):
        self.app = app
        self.gate = gate or admission

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") in {"/health", "/api/health"}:
            await self.app(scope, receive, send)
            return
        if not self.gate.enter():
            await send(
                {
                    "type": "http.response.start",
                    "status": 503,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"retry-after", b"5"),
                    ],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"detail":"Service is draining for rollout."}',
                }
            )
            return
        try:
            await self.app(scope, receive, send)
        finally:
            self.gate.leave()


async def status(command="status"):
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection("/tmp/kadan-rollout.sock"), 3
    )
    try:
        writer.write(command.encode() + b"\n")
        await writer.drain()
        result = json.loads(
            await asyncio.wait_for(reader.readline(), 30 if command == "barrier" else 3)
        )
        print(json.dumps(result))
        if "error" in result:
            raise SystemExit(1)
    finally:
        writer.close()
        await writer.wait_closed()


SMOKE_TARGETS = {
    "kadan-api": (
        "http://127.0.0.1:8630/api/health",
        "http://tsuchi-api:8000/api/health",
        "http://hikari-api:8000/health",
    ),
    "kadan-runner": (
        "http://tsuchi-api:8000/api/health",
        "http://hikari-api:8000/health",
    ),
    "tsuchi-api": ("http://127.0.0.1:8000/api/health",),
    "hikari-api": ("http://127.0.0.1:8000/health",),
}


def smoke():
    """Read-only application and dependency checks; never submit inference."""
    targets = SMOKE_TARGETS.get(os.getenv("KADAN_ROLLOUT_SERVICE", ""))
    if targets is None:
        raise ValueError("Service is not enrolled for rollout smoke checks")
    opener = build_opener(ProxyHandler({}))
    for url in targets:
        with opener.open(url, timeout=3) as response:
            if (
                response.status != 200
                or json.loads(response.read(65536)).get("status") != "ok"
            ):
                raise ValueError("Rollout health check failed")
    return {"ready": True}


if __name__ == "__main__":
    if sys.argv[1:] == ["status"]:
        asyncio.run(status())
    elif sys.argv[1:] == ["barrier"]:
        asyncio.run(status("barrier"))
    elif sys.argv[1:] == ["smoke"]:
        print(json.dumps(smoke()))
    else:
        raise SystemExit(
            "Usage: python -m kadan_tracing.rollout {status|barrier|smoke}"
        )
