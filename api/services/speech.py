"""Provider-neutral WAV transport using the API process's shared resource owner."""
import base64
import threading

from api.inference.speech import SpeechInput, SpeechRuntime, SpeechUnavailable, wav_duration
from api.inference.resources import ResourceBusy, ResourceCancelled, ResourceExhausted
from api.services.speech_providers import speech_registry
from api.services.runtime import runtime_manager

speech_runtime = SpeechRuntime(speech_registry, runtime_manager.ensure_resources)


def validate_request(request: dict) -> str:
    """Resolve and validate through the registered adapter, without allocating."""
    normalized, _ = speech_runtime.registry.resolve(SpeechInput(**request))
    return normalized.model_id


def speech_models():
    return speech_runtime.registry.models()


def generate_speech(request: dict, cancel: threading.Event) -> dict:
    try:
        result = speech_runtime.generate(SpeechInput(**request), cancel)
    except (SpeechUnavailable, ResourceBusy, ResourceCancelled, ResourceExhausted):
        raise
    except Exception as exc:
        raise SpeechUnavailable('Speech provider failed; any incomplete cleanup retains resource ownership.') from exc
    duration = wav_duration(result)
    return dict(voice=result.voice, meta='WAV', script=request['script'],
                time=f'{duration:.1f}s', audio_base64=base64.b64encode(result.wav).decode('ascii'),
                mime_type='audio/wav')
