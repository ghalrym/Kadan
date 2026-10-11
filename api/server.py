import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from fastapi import FastAPI
from api.routes import health
from api.routes.v1 import decisions, images, metrics, requests, settings, videos, models
from api.routes.v1.audio import speech, transcriptions
from api.routes.v1.chat import completions, messages
from api.routes.v1.images import edits, generations as image_generations
from api.routes.v1.videos import generations as video_generations
from api.routes import model_lifecycle
from api.services.model_downloads import model_manager
from api.services.telemetry import TelemetryMiddleware
from api.services.inference import inference


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the FIFO; selected models prepare on requests, never on API reload."""
    async with AsyncExitStack() as cleanup:
        cleanup.push_async_callback(asyncio.to_thread, model_manager.close)
        cleanup.push_async_callback(inference.close)
        await inference.start()
        yield

app = FastAPI(
    lifespan=lifespan,
    title="Kadan API", version="0.0.1",
    description="Local model downloads and selection are persisted. One native C++ worker owns the bounded FIFO, model engines, resource admission and lifecycle. The API submits requests and presents native results. Chat history is client-owned.",
)

# Observe only generation POST handlers; dashboard polling is excluded.
app.add_middleware(TelemetryMiddleware)

for router in (
    health.router, messages.router, completions.router, decisions.router,
    images.router, image_generations.router, edits.router,
    video_generations.router, videos.router, speech.router, transcriptions.router,
    metrics.router, requests.router, settings.router, models.router, model_lifecycle.router,
):
    app.include_router(router)
