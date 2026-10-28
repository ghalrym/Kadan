"""Standalone speech synthesis through the fixed LFM3 audio worker."""

from typing import Literal

from experimental_worker.speech_voice import (
    DEFAULT_GENDER,
    DEFAULT_LANGUAGE,
    GENDERS,
    LANGUAGES,
    SpeechGender,
    SpeechLanguage,
)
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from api.pydantic_models.media import GeneratedSpeech
from api.services.inference import inference
from api.services.inference_errors import InferenceFailure
from api.services.inference_http import infer

router = APIRouter(prefix="/v1/audio/speech", tags=["Audio"])


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    script: str = Field(min_length=1)
    model_id: Literal["lfm3-audio-2b-bf16"] = "lfm3-audio-2b-bf16"
    language: SpeechLanguage = DEFAULT_LANGUAGE
    gender: SpeechGender = DEFAULT_GENDER


class SpeechResponse(BaseModel):
    audio: GeneratedSpeech


class SpeechHistoryResponse(BaseModel):
    audio: list[GeneratedSpeech]
    script: str


class SpeechUnavailable(BaseModel):
    detail: str


class SpeechModelOption(BaseModel):
    id: Literal["lfm3-audio-2b-bf16"]
    name: str
    languages: list[SpeechLanguage]
    genders: list[SpeechGender]
    default_language: SpeechLanguage = DEFAULT_LANGUAGE
    default_gender: SpeechGender = DEFAULT_GENDER


@router.get("/models", operation_id="listSpeechModels")
def list_speech_models() -> list[SpeechModelOption]:
    """Expose the fixed audio checkpoint's supported language and gender choices."""
    try:
        models = inference.models_status().models
    except InferenceFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    return [
        SpeechModelOption(
            id=model.id,
            name=model.display_name or model.id,
            languages=list(LANGUAGES),
            genders=list(GENDERS),
        )
        for model in models
        if model.id == "lfm3-audio-2b-bf16"
        and model.kind == "audio"
        and model.inference_available
    ]


@router.get("", operation_id="listSpeech")
def list_speech() -> SpeechHistoryResponse:
    return SpeechHistoryResponse(audio=[], script="")


@router.post(
    "",
    operation_id="generateSpeech",
    responses={
        503: {"model": SpeechUnavailable, "description": "Speech provider unavailable"}
    },
)
async def generate_speech(body: SpeechRequest, request: Request) -> SpeechResponse:
    """Queue the approved LFM checkpoint; cancellation waits for owned cleanup."""
    selected = body.model_copy(update={"model_id": body.model_id})
    result = await infer(request, inference.submit(selected, feature="tts"))
    return SpeechResponse(audio=result)
