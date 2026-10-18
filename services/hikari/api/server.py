import logging
from contextlib import AsyncExitStack, asynccontextmanager
from fastapi import FastAPI
from api.routes import health
from api.routes.v1 import (
    decisions,
    images,
    metrics,
    models,
    requests,
    settings,
    videos,
    workers,
)
from api.routes.v1.audio import speech, transcriptions
from api.routes.v1.chat import completions, messages
from api.routes.v1.images import edits, generations as image_generations
from api.routes.v1.images import requests as image_requests
from api.routes.v1.videos import generations as video_generations
from api.routes import model_lifecycle
from api.services.telemetry import TelemetryMiddleware
from api.services.inference import inference


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the worker hub and local machine; models load on requests, never on API reload."""
    async with AsyncExitStack() as cleanup:
        cleanup.push_async_callback(inference.close)
        await inference.start()
        yield


# Uvicorn configures only its own loggers; scheduling decisions log at INFO.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)

app = FastAPI(
    lifespan=lifespan,
    title="Hikari API",
    version="0.0.1",
    description="Local model downloads and selection are persisted. Each machine runs one native C++ process with a worker per inference type; workers connect to /v1/workers and own their engines and queues. The API grants every worker its memory budget by priority, routes requests, and presents native results. Chat history is client-owned.",
)

# Observe only generation POST handlers; dashboard polling is excluded.
app.add_middleware(TelemetryMiddleware)

for router in (
    health.router,
    messages.router,
    completions.router,
    decisions.router,
    images.router,
    image_generations.router,
    edits.router,
    image_requests.router,
    video_generations.router,
    videos.router,
    speech.router,
    transcriptions.router,
    metrics.router,
    requests.router,
    settings.router,
    models.router,
    model_lifecycle.router,
    workers.router,
):
    app.include_router(router)
