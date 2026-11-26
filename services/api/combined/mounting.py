"""Mount separate FastAPI services under prefixes and run their lifespans."""

from collections.abc import Mapping
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from kadan_tracing import configure, shutdown


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
    return app
