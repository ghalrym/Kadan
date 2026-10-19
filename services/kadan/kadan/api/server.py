import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from kadan_tracing import RequestTracing, configure, shutdown
from kadan_tracing.rollout import RolloutAdmission, admission

from .config import ServiceError, Settings
from .http_safety import LocalRequestSafety
from .routes import agents, health, runners, status
from .routes.agents import assets, chain, pets, resources
from .routes.agents.assets import content
from .routes.agents.conversations import notes, runs
from .services import LocalServices


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        provider = configure("kadan-api")
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120, connect=3),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            app.state.services = LocalServices(client, config)
            try:
                async with admission.observe():
                    yield
            finally:
                await shutdown(provider)

    app = FastAPI(title="Kadan", version="0.1.0", lifespan=lifespan)
    app.add_middleware(RequestTracing)
    app.add_middleware(RolloutAdmission)
    app.add_middleware(LocalRequestSafety, allowed_hosts=config.allowed_hosts)

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, error: ServiceError):
        return JSONResponse(
            {"detail": str(error), "code": error.code}, status_code=error.status
        )

    for router in (
        health.router,
        runners.router,
        status.router,
        agents.router,
        resources.router,
        chain.router,
        pets.router,
        assets.router,
        content.router,
        notes.router,
        runs.router,
    ):
        app.include_router(router)

    frontend = Path(config.frontend_directory)
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


logging.basicConfig(level=logging.INFO)

app = create_app()
