import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from kadan_tracing import RequestTracing, configure, shutdown

from .config import ServiceError, Settings
from .routes import (
    agents,
    calls,
    health,
    mcp,
    runners,
    status,
    transcriptions,
    updates,
    work,
)
from .routes.agents import (
    assets,
    conversations,
    memories,
    pets,
    resources,
    workspace,
)
from .routes.agents.assets import content
from .routes.agents.conversations import notes, results, runs
from .routes.agents.space import folders, items, pages, slides, trash
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
                yield
            finally:
                await shutdown(provider)

    app = FastAPI(title="Kadan", version="0.1.0", lifespan=lifespan)
    app.add_middleware(RequestTracing)

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, error: ServiceError):
        return JSONResponse(
            {"detail": str(error), "code": error.code}, status_code=error.status
        )

    for router in (
        health.router,
        runners.router,
        status.router,
        updates.router,
        mcp.router,
        work.router,
        transcriptions.router,
        calls.router,
        agents.router,
        resources.router,
        workspace.router,
        memories.router,
        conversations.router,
        pets.router,
        assets.router,
        content.router,
        notes.router,
        results.router,
        runs.router,
        items.router,
        trash.router,
        folders.router,
        pages.router,
        slides.router,
    ):
        app.include_router(router)
    return app


logging.basicConfig(level=logging.INFO)

app = create_app()
