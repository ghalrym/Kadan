"""Sequence Whisper and S1-mini through the same worker and resource ledger."""

from pydantic import BaseModel, ValidationError

from api.pydantic_models.inference_task import TranscriptionFormattingTask
from api.services.inference import inference
from api.services.inference_errors import InferenceFailure


async def transcribe(body: BaseModel) -> dict:
    result = await inference.submit(body, feature="stt")
    if result["formatting_status"] == "disabled":
        return result
    if not result["raw_text"].strip():
        return {**result, "formatting_status": "empty"}
    try:
        formatted = await inference.submit(
            TranscriptionFormattingTask(text=result["raw_text"]),
            feature="stt_formatting",
        )
    except (InferenceFailure, ValidationError) as exc:
        return {
            **result,
            "text": result["raw_text"],
            "formatting_status": "failed",
            "formatting_error": str(exc),
        }
    return {
        **result,
        "text": formatted["text"],
        "formatting_status": "formatted",
        "formatting_model": formatted["model"],
    }
