import asyncio

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter()


class TranscriptionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Twenty seconds of mono 16-kHz PCM, including base64 and the WAV header.
    audio: str = Field(max_length=900_000, pattern=r"^data:audio/wav;base64,.+")


class TranscriptionResult(BaseModel):
    text: str = Field(max_length=16_000)


@router.post("/api/transcriptions")
async def transcribe(
    value: TranscriptionInput, request: Request
) -> TranscriptionResult:
    async def disconnect():
        while (await request.receive())["type"] != "http.disconnect":
            pass

    upstream = asyncio.create_task(
        request.app.state.services.json_request(
            "Hikari",
            "POST",
            "/v1/audio/transcriptions",
            json={"audio": value.audio, "formatting": False},
            timeout=120,
        )
    )
    watcher = asyncio.create_task(disconnect())
    try:
        done, _ = await asyncio.wait(
            (upstream, watcher), return_when=asyncio.FIRST_COMPLETED
        )
        if upstream not in done:
            raise HTTPException(499, "Client disconnected; transcription cancelled.")
        return TranscriptionResult.model_validate(await upstream)
    finally:
        # Cancelling HTTPX closes its upstream connection, letting Hikari cancel work.
        for task in (upstream, watcher):
            if not task.done():
                task.cancel()
        await asyncio.gather(upstream, watcher, return_exceptions=True)
