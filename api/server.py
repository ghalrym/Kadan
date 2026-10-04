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
    description="Local model downloads and selection are persisted. Chat uses Kadan's explicitly loaded inference adapter and returns an error when no model is ready or the checkpoint is unsupported. Decisions use a separate resident CPU Laya specialist without requiring a loaded chat model. Other generation, history and metrics endpoints remain mock fixtures; media is placeholder metadata. This is not an OpenAI-compatible API.",
)

for router in (
    health.router, messages.router, completions.router, decisions.router,
    images.router, image_generations.router, edits.router,
    video_generations.router, videos.router, speech.router, transcriptions.router,
    metrics.router, requests.router, settings.router, models.router, model_lifecycle.router,
):
    app.include_router(router)
