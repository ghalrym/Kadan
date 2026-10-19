"""CPU-only protocol fixture. Never imports Kadan, Tsuchi, or inference engines."""

import asyncio
import hashlib
import json
import os
import signal
import sys
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.request import ProxyHandler, build_opener
from uuid import uuid4

from kadan_tracing.rollout import admission

SERVICE = os.environ["FIXTURE_SERVICE"]
CONTROL = Path("/control")
DATA = Path("/verification")
OWNER = str(uuid4())
LEASE = DATA / "runner.json"
PORT = (
    8630
    if SERVICE == "kadan-api"
    else 8000
    if SERVICE in {"tsuchi-api", "hikari-api"}
    else 8080
)


def health():
    return (
        Path("/tmp/fixture-ready").exists()
        and not (CONTROL / f"unhealthy-{SERVICE}").exists()
    )


def observation():
    if SERVICE == "kadan-runner":
        return {
            "runner": {"runner_id": "fixture-runner", "worker_id": OWNER, "capacity": 5}
        }
    if SERVICE == "tsuchi-api":
        return {
            "runner_leases": {
                "observed_at": datetime.now(UTC).isoformat(),
                "runners": [json.loads(LEASE.read_text())] if LEASE.exists() else [],
            }
        }
    if SERVICE == "hikari-api":
        return {"api_generation": OWNER}
    return {}


async def work(stop):
    admitted = False
    hold = CONTROL / f"hold-{SERVICE}"
    try:
        while not stop.is_set():
            if SERVICE == "kadan-runner" and not admission.fenced:
                row = json.loads(LEASE.read_text()) if LEASE.exists() else None
                now = datetime.now(UTC)
                if (
                    row is None
                    or row["worker_id"] == OWNER
                    or datetime.fromisoformat(row["lease_expires_at"]) <= now
                ):
                    # The fixture TTL is deliberately short; the controller reads it,
                    # rather than assuming a production timeout.
                    row = {
                        "runner_id": "fixture-runner",
                        "worker_id": OWNER,
                        "capacity": 5,
                        "assigned_agent_count": 0,
                        "lease_expires_at": (now + timedelta(seconds=5)).isoformat(),
                    }
                    temporary = LEASE.with_suffix(".tmp")
                    temporary.write_text(json.dumps(row))
                    temporary.replace(LEASE)
            if hold.exists() and not admitted:
                admitted = admission.enter()
            if admitted and not hold.exists():
                admission.leave()
                admitted = False
            await asyncio.sleep(0.05)
    finally:
        if admitted:
            admission.leave()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"status":"ok"}'
        self.send_response(200 if health() else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


async def serve():
    DATA.mkdir(exist_ok=True)
    try:
        with (DATA / "sentinel").open("x") as stream:
            stream.write(os.environ["FIXTURE_SENTINEL"])
    except FileExistsError:
        pass
    admission.extra_status = observation
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        async with admission.observe():
            Path("/tmp/fixture-ready").write_text("ready")
            await work(stop)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    if sys.argv[1:] == ["serve"]:
        asyncio.run(serve())
    elif sys.argv[1:] == ["health"]:
        raise SystemExit(0 if health() else 1)
    elif sys.argv[1:] == ["inspect"]:
        paths = {
            "instruction": Path("/app/AGENT.md"),
            "volume": DATA / "sentinel",
            "bind": Path("/data/sentinel"),
        }
        print(
            json.dumps(
                {
                    key: hashlib.sha256(path.read_bytes()).hexdigest()
                    for key, path in paths.items()
                    if path.exists()
                }
            )
        )
    elif sys.argv[1:2] == ["fetch"]:
        with build_opener(ProxyHandler({})).open(sys.argv[-1], timeout=3) as response:
            response.read()
    else:
        raise SystemExit("Expected serve, health, inspect, or fetch")
