"""Voice turns reuse the native FIFO; browser listening is independent of inference."""

import base64
import binascii
import io
import time
import wave
from dataclasses import asdict, dataclass
from typing import Literal

from api.pydantic_models.chat import ChatMessage
from api.pydantic_models.inference_task import ChatTask, SpeechTask, TranscriptionTask
from api.pydantic_models.model_store import TierId
from api.services.inference import inference
from api.services.inference_errors import InferenceFailure

SpeechBackend = Literal["current", "drama3", "drama3-int4"]


@dataclass(frozen=True)
class ConversationModels:
    tier: TierId
    asr: str
    chat: str
    speech: str | None


def conversation_models(
    service, expected_tier: TierId | None = None
) -> ConversationModels:
    snapshot = service.models_status()
    if expected_tier is not None and snapshot.selected_tier != expected_tier:
        raise InferenceFailure(
            "The active tier changed. Stop and restart the conversation.", 409
        )
    tier = next(
        (item for item in snapshot.tiers if item.id == snapshot.selected_tier), None
    )
    if tier is None:
        raise InferenceFailure("Select a hardware tier in Settings first.", 409)
    if not tier.models.transcription:
        raise InferenceFailure("The active tier has no transcription model.", 409)
    return ConversationModels(
        tier.id, tier.models.transcription, tier.models.llm, tier.models.speech
    )


def installed_model(service, model_id: str | None) -> str | None:
    model = next(
        (item for item in service.models_status().models if item.id == model_id), None
    )
    if model is None:
        return f"Model {model_id or '(unassigned)'} is unavailable in this worker."
    if not model.inference_available:
        return f"This worker does not support inference for {model.id}."
    if model.status != "complete":
        setup = (
            "through POST /v1/models/drama-3/download (see the Voice conversation README)"
            if model.id == "drama-3"
            else "in Settings"
        )
        return f"Install {model.id} {setup} before starting. Conversations never download models."
    return None


@dataclass(frozen=True)
class SpeechBackendAvailability:
    id: SpeechBackend
    name: str
    speech_model: str | None
    available: bool
    reason: str | None
    streaming: bool


def backend_availability(
    service,
    backend: SpeechBackend,
    selection: ConversationModels | None,
    reason: str | None,
) -> SpeechBackendAvailability:
    if backend == "current":
        name, model, streaming = (
            "Current tier TTS",
            selection.speech if selection else None,
            False,
        )
        if model != "qwen-tts-1.7b-custom":
            reason = reason or "The active tier has no supported current speech model."
    elif backend == "drama3":
        name, model, streaming = "Drama 3 FP8 · experimental native", "drama-3", True
    else:
        name, model, streaming = "Drama 3 INT4 · unsupported", "drama-3-int4", True
        reason = "The native worker does not implement Drama 3 GPTQ INT4 weights. No fallback is used."
    reason = reason or installed_model(service, model)
    return SpeechBackendAvailability(
        backend, name, model, reason is None, reason, streaming
    )


def capabilities(service=inference) -> dict:
    selection = None
    reason = None
    try:
        selection = conversation_models(service)
    except InferenceFailure as exc:
        reason = str(exc)
    if selection:
        reason = installed_model(service, selection.asr) or installed_model(
            service, selection.chat
        )

    return {
        "available": reason is None,
        "reason": reason,
        "tier": selection.tier if selection else None,
        "asr_model": selection.asr if selection else None,
        "chat_model": selection.chat if selection else None,
        "scheduling": "Serial native inference; continuous browser capture and playback",
        "backends": [
            asdict(backend_availability(service, backend, selection, reason))
            for backend in ("current", "drama3", "drama3-int4")
        ],
        "notes": [
            "ASR uses bounded Whisper windows with revised interim transcripts, not an incremental model session.",
            "Chat finishes before speech synthesis starts. Changing modality may reload weights; timings include queue/load time.",
            "The worker admits each request against host and individual GPU headroom. Additional GPUs do not enable simultaneous ASR/chat/TTS.",
            "Drama 3 uses the experimental native FP8-weight/F32-compute implementation. Published SGLang latency does not apply.",
        ],
    }


def require_backend(
    service, tier: TierId, backend: SpeechBackend
) -> tuple[ConversationModels, str]:
    selection = conversation_models(service, tier)
    reason = installed_model(service, selection.asr) or installed_model(
        service, selection.chat
    )
    option = backend_availability(service, backend, selection, reason)
    if option.reason:
        raise InferenceFailure(option.reason, 409)
    assert option.speech_model is not None
    return selection, option.speech_model


async def transcribe_window(
    audio: str, tier: TierId, model: str, service=inference
) -> dict:
    selection = conversation_models(service, tier)
    if model != selection.asr:
        raise InferenceFailure("Transcription must use the active tier's model.", 409)
    if reason := installed_model(service, selection.asr):
        raise InferenceFailure(reason, 409)
    # Bypass optional formatting: its additional model is not part of recognition.
    result = await service.submit(
        TranscriptionTask(
            audio=audio,
            model=selection.asr.removeprefix("whisper-"),
            formatting=False,
            expected_tier=tier,
            require_installed=True,
        ),
        feature="stt",
    )
    return {"text": result["raw_text"], "model": selection.asr}


def buffered_pcm(result: dict):
    """Frame the unchanged Qwen WAV result for the same browser PCM player."""
    try:
        encoded = result["audio_base64"]
        if len(encoded) > 6 * 1024**2:
            raise ValueError("Speech exceeds the buffered audio limit")
        audio = base64.b64decode(encoded, validate=True)
        with wave.open(io.BytesIO(audio), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (
                1,
                2,
                "NONE",
            ):
                raise ValueError("Expected mono 16-bit PCM")
            rate = wav.getframerate()
            if not 8000 <= rate <= 96000 or wav.getnframes() < 1:
                raise ValueError("Invalid speech sample rate or empty audio")
            remaining = wav.getnframes()
            while remaining:
                frames = min(rate, remaining)
                chunk = wav.readframes(frames)
                if len(chunk) != frames * 2:
                    raise ValueError("Truncated speech WAV")
                remaining -= frames
                yield chunk, rate
    except (
        KeyError,
        TypeError,
        ValueError,
        binascii.Error,
        wave.Error,
        EOFError,
    ) as exc:
        raise InferenceFailure(f"Invalid worker speech audio: {exc}", 502) from exc


async def conversation_turn(
    tier: TierId,
    backend: SpeechBackend,
    messages: list[ChatMessage],
    speaker: str,
    service=inference,
):
    selection, speech_model = require_backend(service, tier, backend)
    started = time.monotonic()
    first_token = None
    stream = None
    text = ""
    finish = None
    try:
        stream, actual_model = await service.open_chat_stream(
            ChatTask(
                messages=messages,
                model=selection.chat,
                stream=True,
                reuse_prefix=False,
                expected_tier=tier,
                require_installed=True,
            )
        )
        if actual_model != selection.chat:
            raise InferenceFailure(
                "Worker selected a different conversation model.", 502
            )
        async for event in stream:
            if content := event.get("content"):
                if first_token is None:
                    first_token = (time.monotonic() - started) * 1000
                text += content
                if len(text.encode("utf-8")) > 8000:
                    raise InferenceFailure("Voice response exceeded 8000 bytes.", 413)
                yield {"type": "text", "text": content}
            if "finish_reason" in event:
                finish = event["finish_reason"]
    finally:
        if stream is not None:
            await stream.aclose()
    if finish != "stop" or not text.strip():
        raise InferenceFailure(
            "Chat ended without a complete spoken response. Shorten the request.", 502
        )
    # Recheck between jobs; native expected_tier also closes the queued-setting race.
    require_backend(service, tier, backend)
    speech_started = time.monotonic()
    first_audio = None
    duration = 0.0
    stream = None
    speech = SpeechTask(
        script=text,
        model_id=speech_model,
        language="Auto",
        voice={
            "mode": "custom",
            "speaker": speaker if backend == "current" else "default",
        },
        stream=backend != "current",
        expected_tier=tier,
        require_installed=True,
    )
    native_timing = None
    try:
        if backend == "current":
            result = await service.submit(speech, feature="tts")
            for pcm, rate in buffered_pcm(result):
                if first_audio is None:
                    first_audio = time.monotonic()
                duration += len(pcm) / (2 * rate)
                yield {
                    "type": "audio",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                    "sample_rate": rate,
                }
        else:
            stream = await service.open_speech_stream(speech)
            stopped = False
            async for event in stream:
                if "audio" in event:
                    try:
                        pcm = base64.b64decode(event["audio"], validate=True)
                    except (ValueError, binascii.Error) as exc:
                        raise InferenceFailure(
                            "Invalid native PCM encoding.", 502
                        ) from exc
                    rate = event["sample_rate"]
                    if rate != 44100 or not pcm or len(pcm) % 2:
                        raise InferenceFailure("Invalid Drama 3 PCM format.", 502)
                    if first_audio is None:
                        first_audio = time.monotonic()
                    duration += len(pcm) / (2 * rate)
                    if duration > 60:
                        raise InferenceFailure(
                            "Speech exceeded the 60-second benchmark limit.", 413
                        )
                    yield {
                        "type": "audio",
                        "audio": event["audio"],
                        "sample_rate": rate,
                    }
                if "finish_reason" in event:
                    stopped = event["finish_reason"] == "stop"
                    native_timing = event.get("timing")
            if not stopped:
                raise InferenceFailure(
                    "Speech ended without a complete native terminal event.", 502
                )
    finally:
        if stream is not None:
            await stream.aclose()
    if first_audio is None or duration <= 0:
        raise InferenceFailure("Speech returned no audio.", 502)
    ended = time.monotonic()
    yield {
        "type": "done",
        "tier": tier,
        "backend": backend,
        "chat_model": selection.chat,
        "speech_model": speech_model,
        "timing": {
            "llm_first_token_ms": first_token,
            "speech_first_audio_ms": (first_audio - speech_started) * 1000,
            "turn_first_audio_ms": (first_audio - started) * 1000,
            "audio_duration_seconds": duration,
            "speech_rtf": (ended - speech_started) / duration,
            "turn_ms": (ended - started) * 1000,
            "native": native_timing,
        },
    }
