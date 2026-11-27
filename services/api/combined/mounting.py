"""Mount separate FastAPI services under prefixes and run their lifespans."""

from collections.abc import Mapping
from contextlib import AsyncExitStack, asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from kadan_tracing import configure, shutdown


_API_PREFIXES = (
    "/.well-known",
    "/kadan/.well-known",
    "/kadan/api",
    "/kadan/mcp",
    "/kadan/docs",
    "/kadan/openapi.json",
    "/tsuchi/api",
    "/tsuchi/docs",
    "/tsuchi/openapi.json",
    "/hikari/v1",
    "/hikari/model-lifecycle",
    "/hikari/health",
    "/hikari/docs",
    "/hikari/openapi.json",
)

_KADAN_WELL_KNOWN_PREFIXES = (
    "/.well-known/oauth-protected-resource",
    "/.well-known/oauth-authorization-server",
    "/.well-known/openid-configuration",
)


def _is_api_path(path: str) -> bool:
    return any(
        path == prefix or path.startswith(f"{prefix}/") for prefix in _API_PREFIXES
    )


def _is_kadan_well_known_path(path: str) -> bool:
    return any(
        path == prefix or path.startswith(f"{prefix}/")
        for prefix in _KADAN_WELL_KNOWN_PREFIXES
    )


class _KadanWellKnownMiddleware:
    """Forward root RFC 9728 / RFC 8414 OAuth discovery paths to the mounted Kadan app."""

    def __init__(self, app, kadan_app: FastAPI):
        self.app = app
        self.kadan_app = kadan_app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and _is_kadan_well_known_path(scope.get("path", "")):
            base_root = str(scope.get("root_path") or "").rstrip("/")
            forwarded_scope = {**scope, "root_path": f"{base_root}/kadan"}
            await self.kadan_app(forwarded_scope, receive, send)
            return
        await self.app(scope, receive, send)


def lifespan_for(apps: Mapping[str, FastAPI]):
    """Run mounted apps' lifespans, which Starlette does not do for mounts.

    Apps start in mapping order and stop in reverse. One tracer provider serves
    the process, so the services' own tracing setup becomes a no-op.
    """

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        provider = configure("kadan-services")
        try:
            async with AsyncExitStack() as stack:
                for app in apps.values():
                    await stack.enter_async_context(app.router.lifespan_context(app))
                yield
        finally:
            await shutdown(provider)

    return lifespan


def _wrap_frontend_spa(app: FastAPI, frontend_root: Path):
    """Serve built unified frontend for browser routes without intercepting API prefixes."""

    async def asgi(scope, receive, send):
        if scope["type"] == "http" and scope.get("method") in {"GET", "HEAD"}:
            path = scope.get("path", "/")
            if not _is_api_path(path):
                rel = path.lstrip("/")
                if rel:
                    candidate = (frontend_root / rel).resolve()
                    if candidate.is_file() and frontend_root in candidate.parents:
                        response = FileResponse(candidate)
                        await response(scope, receive, send)
                        return
                index = frontend_root / "index.html"
                if index.is_file():
                    response = FileResponse(index)
                    await response(scope, receive, send)
                    return
                raise HTTPException(status_code=404, detail="Frontend not built")
        await app(scope, receive, send)

    return asgi


def create_app(apps: Mapping[str, FastAPI]) -> FastAPI:
    app = FastAPI(
        title="Kadan services",
        lifespan=lifespan_for(apps),
        # Each service publishes its own reference under its prefix, e.g. /hikari/docs.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    for name, service in apps.items():
        app.mount(f"/{name}", service, name=name)
    if "kadan" in apps:
        app.add_middleware(_KadanWellKnownMiddleware, kadan_app=apps["kadan"])
    frontend_dir = os.environ.get("KADAN_FRONTEND_DIR", "").strip()
    if frontend_dir:
        root = Path(frontend_dir).resolve()
        if root.is_dir():
            return _wrap_frontend_spa(app, root)
    return app
