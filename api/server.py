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
from api.services.runtime import runtime_manager
from api.services.telemetry import TelemetryMiddleware
from api.services.decisions import decision_manager
from api.services.video_jobs import video_jobs
from api.memory_manager import memory_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Restore the selected model on startup and release inference before downloads on shutdown."""
    async with AsyncExitStack() as cleanup:
        cleanup.push_async_callback(asyncio.to_thread, model_manager.close)
        cleanup.push_async_callback(runtime_manager.close)
        cleanup.push_async_callback(decision_manager.close)
        cleanup.push_async_callback(asyncio.to_thread, video_jobs.close)
        cleanup.push_async_callback(memory_manager.close)
        if await memory_manager.start():
            await runtime_manager.start()
        yield

app = FastAPI(
    lifespan=lifespan,
    title="Kadan API", version="0.0.1",
    description="Local model downloads and selection are persisted. Inference uses a bounded Redis queue consumed inside the API process. Chat uses Kadan's selected native adapter; Decisions use a resident CPU Laya specialist. H3 video and Whisper transcription use shared memory admission. Image and speech providers are unavailable. Monitoring reports bounded process-local HTTP telemetry and observed memory. Chat history is client-owned. This is not an OpenAI-compatible API.",
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
