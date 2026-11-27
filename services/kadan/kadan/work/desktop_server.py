"""Local Desktop Work sidecar server & WorkRunner for Kadan Desktop.

Runs on the user's desktop computer (spawned by `kadan-desktop`) so that:
- Project folder browsing and validation inspect the user's local computer.
- Coding harnesses (Codex, Claude Code, Antigravity, Grok, and Inkling tool loops)
  execute locally on the user's computer against their local folders and CLI sign-ins.
- Durable Work state (projects, threads, runs, activity, saved harness settings)
  and Inkling 2 Mini inference talk to the active Kadan Server's Tsuchi and Hikari APIs.
"""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..api.config import ServiceError, Settings
from ..api.routes import work as work_routes
from ..api.services import LocalServices
from .runner import WorkRunner

logger = logging.getLogger(__name__)


class UpstreamUpdateRequest(BaseModel):
    active_host: str


def _ensure_cli_paths_in_env() -> None:
    """Include common user CLI directories when launched from a GUI/systemd session."""
    home = Path.home()
    extra = [
        home / ".local" / "share" / "mise" / "shims",
        home / ".local" / "bin",
        home / ".cargo" / "bin",
        Path("/usr/local/bin"),
    ]
    current = os.environ.get("PATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    seen = set(parts)
    for candidate in extra:
        text = str(candidate)
        if candidate.is_dir() and text not in seen:
            parts.append(text)
            seen.add(text)
    if parts:
        os.environ["PATH"] = os.pathsep.join(parts)


def derive_upstream_urls(active_host: str) -> tuple[str, str]:
    base = active_host.strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        base = f"http://{base}"
    return f"{base}/tsuchi", f"{base}/hikari"


def create_desktop_work_app(
    settings: Settings | None = None,
    *,
    start_runner: bool = True,
    http_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owns_client = http_client is None
        client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(120, connect=5),
            follow_redirects=False,
            trust_env=False,
        )
        services = LocalServices(client, config)
        runner = WorkRunner(services)
        app.state.services = services
        app.state.work_runner = runner
        app.state.start_runner = start_runner
        if start_runner:
            runner.start()
        try:
            yield
        finally:
            await runner.close()
            if owns_client:
                await client.aclose()

    app = FastAPI(title="Kadan Desktop Work", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, error: ServiceError):
        return JSONResponse(
            {"detail": str(error), "code": error.code}, status_code=error.status
        )

    @app.get("/api/work/health")
    async def work_health(request: Request):
        services: LocalServices = request.app.state.services
        return {
            "status": "ok",
            "mode": "desktop",
            "tsuchi_url": services.settings.tsuchi_url,
            "hikari_url": services.settings.hikari_url,
        }

    @app.put("/api/work/upstream")
    async def update_upstream(request: Request, body: UpstreamUpdateRequest):
        tsuchi_url, hikari_url = derive_upstream_urls(body.active_host)
        services: LocalServices = request.app.state.services
        old_runner: WorkRunner = request.app.state.work_runner
        if (
            services.settings.tsuchi_url == tsuchi_url
            and services.settings.hikari_url == hikari_url
        ):
            return {
                "status": "ok",
                "tsuchi_url": tsuchi_url,
                "hikari_url": hikari_url,
            }
        await old_runner.close()
        services.settings.tsuchi_url = tsuchi_url
        services.settings.hikari_url = hikari_url
        new_runner = WorkRunner(services)
        request.app.state.work_runner = new_runner
        if request.app.state.start_runner:
            new_runner.start()
        return {
            "status": "ok",
            "tsuchi_url": tsuchi_url,
            "hikari_url": hikari_url,
        }

    app.include_router(work_routes.router)
    return app


def main() -> None:
    _ensure_cli_paths_in_env()
    parser = argparse.ArgumentParser(description="Kadan Desktop local Work sidecar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("KADAN_WORK_PORT", "8645"))
    )
    parser.add_argument(
        "--active-host", default=os.environ.get("KADAN_ACTIVE_HOST", "")
    )
    args = parser.parse_args()

    kwargs: dict[str, str] = {}
    if args.active_host.strip():
        tsuchi_url, hikari_url = derive_upstream_urls(args.active_host)
        kwargs["tsuchi_url"] = tsuchi_url
        kwargs["hikari_url"] = hikari_url
    settings = Settings(**kwargs)
    app = create_desktop_work_app(settings, start_runner=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
