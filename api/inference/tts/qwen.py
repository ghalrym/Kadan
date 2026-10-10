"""Native Qwen speech validation and checkpoint preparation; no Python inference."""
import os
from pathlib import Path

from api.inference.tts.speech_runtime import SpeechInput, SpeechModel, SpeechPlan, SpeechUnavailable
from api.services.model_downloads import model_manager
from api.inference.tts.catalog import SPEECH_MODELS, SPEAKERS
from api.inference.tts import enabled as speech_enabled
from api.inference.native_compute import native_compute
from api.inference.native_assets import ensure_assets
from api.inference.tts.native_worker import NativeSpeechSession, PROCESS_BUDGET, verify_tokenizer, validate as validate_native


class QwenSpeechProvider:
    def models(self):
        model = SPEECH_MODELS['qwen-tts-1.7b-custom']
        return (SpeechModel(model.id, model.name, model.mode, SPEAKERS, True, 'Ryan'),)

    def enabled(self, model_id):
        return model_id == 'qwen-tts-1.7b-custom' and model_id in speech_enabled.ENABLED_SPEECH_MODELS

    def validate(self, request: SpeechInput):
        validate_native(request, allow_empty=True)

    def prepare_assets(self, request, resources, cancel):
        if not os.environ.get('KADAN_NATIVE_TTS_TOKENIZER'):
            entry, checkpoint = model_manager.get_checkpoint(request.model_id)
            if entry.revision != SPEECH_MODELS[request.model_id].revision:
                raise SpeechUnavailable('Speech checkpoint revision mismatch.')
            ensure_assets('tts', checkpoint, model_manager.root / 'native/tts', resources, cancel)

    def prepare_placement(self, request, resources, retained=None):
        return self.prepare(request, resources, retained)

    def prepare(self, request, resources=None, retained=None):
        validate_native(request, allow_empty=True)
        model = SPEECH_MODELS[request.model_id]
        resolver = getattr(model_manager, 'get_checkpoint', None)
        if resolver is None:
            raise SpeechUnavailable('The shared checkpoint catalog integration is required for Qwen3-TTS.')
        try:
            entry, checkpoint = resolver(model.id)
        except ValueError as exc:
            raise SpeechUnavailable(str(exc)) from exc
        if entry.revision != model.revision:
            raise SpeechUnavailable('The selected Qwen checkpoint revision does not match the adapter.')
        binary = Path(os.environ.get('KADAN_NATIVE_TTS_WORKER') or '/opt/kadan/bin/kadan-tts-worker')
        tokenizer = Path(os.environ.get('KADAN_NATIVE_TTS_TOKENIZER') or model_manager.root / 'native/tts/tokenizer.json')
        if not binary.is_absolute() or not tokenizer.is_absolute() or not binary.is_file() or not tokenizer.is_file():
            raise SpeechUnavailable('Configure absolute native TTS worker and tokenizer paths.')
        try:
            verify_tokenizer(checkpoint, tokenizer)
        except (OSError, KeyError, ValueError, TypeError) as error:
            raise SpeechUnavailable("Invalid or missing native TTS tokenizer export.") from error
        compute = native_compute('TTS', resources)
        return SpeechPlan(('qwen-native', model.id, model.revision, str(checkpoint), str(binary), str(tokenizer), *compute.identity),
            PROCESS_BUDGET, lambda: NativeSpeechSession(checkpoint, tokenizer, binary, model.name, compute=compute), compute.device_bytes)
