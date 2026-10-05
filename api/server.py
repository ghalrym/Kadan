from fastapi import FastAPI
from api.routes import health
from api.routes.v1 import decisions, images, metrics, requests, settings, videos, models
from api.routes.v1.audio import speech, transcriptions
from api.routes.v1.chat import completions, messages
from api.routes.v1.images import edits, generations as image_generations
from api.routes.v1.videos import generations as video_generations

app = FastAPI(
    title="Kadan API", version="0.0.1",
    description="Local model downloads and selection are persisted. Generation and media endpoints remain deterministic mock responses until a runtime is configured. Media is placeholder metadata, not downloadable files. This is not an OpenAI-compatible API.",
)

for router in (
    health.router, messages.router, completions.router, decisions.router,
    images.router, image_generations.router, edits.router,
    video_generations.router, videos.router, speech.router, transcriptions.router,
    metrics.router, requests.router, settings.router, models.router,
):
    app.include_router(router)
