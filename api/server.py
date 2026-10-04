import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routes import health
from api.routes.v1 import decisions, images, metrics, requests, settings, videos, models
from api.routes.v1.audio import speech, transcriptions
from api.routes.v1.chat import completions, messages
from api.routes.v1.images import edits, generations as image_generations
from api.routes.v1.videos import generations as video_generations
from api.routes import model_lifecycle
from api.services.model_downloads import model_manager
from api.services.runtime import runtime_manager
from api.services.telemetry import TelemetryMiddleware
from api.services.decisions import decision_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On shutdown, release inference ownership first, then cooperatively stop the downloader on a
    worker thread.
    """
    try:
        yield
    finally:
        try:
            await decision_manager.close()
        finally:
            try:
                await runtime_manager.close()
            finally:
                await asyncio.to_thread(model_manager.close)

app = FastAPI(
    lifespan=lifespan,
    title="Kadan API", version="0.0.1",
    description="Local model downloads and selection are persisted. Chat and decisions use Kadan's explicitly loaded inference adapter and return errors when no model is ready or output is invalid. Image, video, speech and transcription providers are not implemented; those generation endpoints return unavailable errors and media history is empty. Monitoring reports bounded process-local HTTP telemetry and observed memory. Chat history is client-owned. This is not an OpenAI-compatible API.",
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
