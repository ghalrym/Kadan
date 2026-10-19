from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from kadan_tracing import RequestTracing, configure, shutdown
from kadan_tracing.rollout import RolloutAdmission, admission
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles

from .config import Settings, StorageError
from .http_safety import LocalRequestSafety
from .repository import Repository
from .routes import create_router
from .services import FileStorage


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    repository = Repository(settings.data_dir)
    files = FileStorage(repository, settings.max_upload_bytes)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        repository.initialize()
        files.initialize()
        provider = configure("tsuchi-api")
        try:
            async with admission.observe():
                yield
        finally:
            await shutdown(provider)

    app = FastAPI(
        title="Tsuchi",
        description="Local storage and retrieval for agent content.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(RequestTracing)
    app.add_middleware(RolloutAdmission)
    app.add_middleware(
        LocalRequestSafety,
        allowed_hosts=settings.allowed_hosts,
        max_body_bytes=settings.max_upload_bytes + 1024 * 1024,
    )

    @app.exception_handler(StorageError)
    async def storage_error(request: Request, error: StorageError):
        return JSONResponse({"detail": error.detail}, status_code=error.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        detail = "; ".join(
            f"{'.'.join(str(part) for part in entry['loc'])}: {entry['msg']}"
            for entry in error.errors()[:5]
        )
        return JSONResponse({"detail": detail}, status_code=422)

    app.include_router(create_router(repository, files))
    if settings.static_dir.is_dir():
        app.mount(
            "/", StaticFiles(directory=settings.static_dir, html=True), name="website"
        )
    return app


app = create_app()
