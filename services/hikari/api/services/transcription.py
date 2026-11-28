"""Queue speech-to-text transcription through the native Qwen3-ASR worker."""

from pydantic import BaseModel

from api.services.inference import inference


async def transcribe(body: BaseModel) -> dict:
    return await inference.submit(body, feature="stt")
