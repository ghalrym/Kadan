"""Qwen-specific validation, checkpoint preparation and in-process resident adapter."""
import base64
import binascii
import gc
import io
import logging
import os
from pathlib import Path
import threading

from api.inference.placement import select_device
from api.inference.resources import ResourceCancelled
from api.inference.tts.speech_runtime import SpeechInput, SpeechModel, SpeechPlan, SpeechResult, SpeechUnavailable
from api.services.model_downloads import model_manager
from api.inference.tts.catalog import SPEECH_MODELS, SPEAKERS, LANGUAGES
from api.inference.tts import enabled as speech_enabled
from api.inference.tts.native_worker import NativeSpeechSession, PROCESS_BUDGET, verify_tokenizer, validate as validate_native


log = logging.getLogger(__name__)


class QwenSpeechProvider:
    def models(self):
        if os.environ.get("KADAN_NATIVE_TTS_WORKER"):
            model = SPEECH_MODELS["qwen-tts-1.7b-custom"]
            return (SpeechModel(model.id, model.name, model.mode, ("Ryan",), False, "Ryan"),)
        return tuple(SpeechModel(item.id, item.name, item.mode,
            SPEAKERS if item.mode == 'custom' else (), item.id == 'qwen-tts-1.7b-custom',
            'Ryan' if item.mode == 'custom' else None)
            for item in SPEECH_MODELS.values())

    def enabled(self, model_id):
        return model_id in speech_enabled.ENABLED_SPEECH_MODELS

    def validate(self, request: SpeechInput):
        if os.environ.get("KADAN_NATIVE_TTS_WORKER"):
            validate_native(request)
            return
        if request.language not in LANGUAGES:
            raise ValueError('Unsupported speech language')
        voice = request.voice
        if voice['mode'] == 'custom':
            if voice['speaker'] not in SPEAKERS:
                raise ValueError('Unsupported speaker')
            if request.model_id == 'qwen-tts-0.6b-custom' and voice.get('instruction'):
                raise ValueError('Instruction control requires the 1.7B CustomVoice model')
        if voice['mode'] == 'clone':
            if not voice.get('speaker_only') and not (voice.get('transcript') or '').strip():
                raise ValueError('Provide the reference transcript or select speaker-only cloning')
            try:
                if not base64.b64decode(voice['sample'], validate=True):
                    raise ValueError('Empty reference audio')
            except binascii.Error as exc:
                raise ValueError('Reference audio must be base64-encoded audio bytes') from exc

    def prepare_placement(self, request, resources, retained=None):
        return self.prepare(request, resources, retained)

    def prepare(self, request, resources=None, retained=None):
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
        if os.environ.get('KADAN_NATIVE_TTS_WORKER'):
            validate_native(request)
            binary = Path(os.environ['KADAN_NATIVE_TTS_WORKER'])
            tokenizer = Path(os.environ.get('KADAN_NATIVE_TTS_TOKENIZER', ''))
            if not binary.is_absolute() or not tokenizer.is_absolute() or not binary.is_file() or not tokenizer.is_file():
                raise SpeechUnavailable('Configure absolute native TTS worker and tokenizer paths.')
            try:
                verify_tokenizer(checkpoint, tokenizer)
            except (OSError, KeyError, ValueError, TypeError) as error:
                raise SpeechUnavailable("Invalid or missing native TTS tokenizer export.") from error
            return SpeechPlan(('qwen-native', model.id, model.revision, str(checkpoint), str(binary), str(tokenizer)),
                PROCESS_BUDGET, lambda: NativeSpeechSession(checkpoint, tokenizer, binary, model.name))
        device = os.environ.get('KADAN_QWEN_TTS_DEVICE', 'auto')
        budget = model.estimated_bytes * 3 + 2 * 1024**3
        if resources is not None:
            credit = None
            if retained is not None and retained.identity[:4] == ('qwen', model.id, model.revision, str(checkpoint)):
                credit = next(iter(retained.device_bytes.items()), None)
            device = select_device(resources, budget, device, allow_cpu=True, retained=credit)
        elif device == 'auto':
            raise SpeechUnavailable('Automatic speech placement requires the shared resource manager.')
        elif device != 'cpu' and not (device.startswith('cuda:') and device[5:].isdigit()):
            raise SpeechUnavailable('Qwen speech device must be auto, cpu or cuda:N.')
        return SpeechPlan(('qwen', model.id, model.revision, str(checkpoint), device),
            budget, lambda: QwenSpeechSession(checkpoint, device, model.name),
            {} if device == 'cpu' else {int(device[5:]): budget})


class QwenSpeechSession:
    """One in-process model, retained until Kadan evicts or explicitly unloads it.

    Cancellation is cooperative at generated-token boundaries. Loading and audio
    decoding finish before cleanup; callers keep admission ownership until then.
    """
    def __init__(self, checkpoint, device, name):
        self.checkpoint, self.device, self.name = checkpoint, device, name
        self.model = None
        self.resident_device = None

    @staticmethod
    def _check_cancel(cancel):
        if cancel.is_set():
            raise ResourceCancelled('Speech generation cancelled')

    def load(self, cancel):
        self._check_cancel(cancel)
        # Defer heavyweight model imports until Kadan has reserved memory.
        import torch
        from qwen_tts import Qwen3TTSModel
        self.model = Qwen3TTSModel.from_pretrained(str(self.checkpoint),
            local_files_only=True, device_map=None,
            dtype=torch.float32 if self.device == 'cpu' else torch.bfloat16,
            attn_implementation='sdpa')
        self.resident_device = 'cpu'
        self.restore(cancel)
        log.info('Qwen TTS model %s loaded once from checkpoint', id(self.model))
        self._check_cancel(cancel)

    def _move(self, device):
        # The pinned Qwen facade and codec facade are not torch.nn.Modules.
        # Move both owned modules and update their explicit input-device fields.
        import torch
        self.model.model.to(device)
        tokenizer = self.model.model.speech_tokenizer
        tokenizer.model.to(device)
        self.model.device = tokenizer.device = torch.device(device)
        self.resident_device = device

    def offload_to_ram(self):
        import torch
        if self.model is not None and self.resident_device != 'cpu':
            self._move('cpu')
            with torch.cuda.device(self.device):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            log.info('Qwen TTS model %s parked in RAM', id(self.model))

    def restore(self, cancel):
        self._check_cancel(cancel)
        if self.model is not None and self.resident_device != self.device:
            self._move(self.device)
            log.info('Qwen TTS model %s restored to %s', id(self.model), self.device)
        self._check_cancel(cancel)

    def generate(self, request, cancel):
        # Audio/model libraries are needed only while an admitted request runs.
        import numpy as np
        import soundfile as sf
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class Cancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                if cancel.is_set():
                    raise ResourceCancelled('Speech generation cancelled')
                return torch.full((input_ids.shape[0],), False,
                                  dtype=torch.bool, device=input_ids.device)

        self._check_cancel(cancel)
        voice = request.voice
        options = dict(text=request.script, language=request.language,
            non_streaming_mode=True, stopping_criteria=StoppingCriteriaList([Cancelled()]))
        audio = waves = None
        try:
            with torch.inference_mode():
                if voice['mode'] == 'describe':
                    waves, rate = self.model.generate_voice_design(**options, instruct=voice['description'])
                elif voice['mode'] == 'custom':
                    waves, rate = self.model.generate_custom_voice(**options,
                        speaker=voice['speaker'], instruct=voice.get('instruction', ''))
                else:
                    raw = base64.b64decode(voice['sample'], validate=True)
                    audio, rate = sf.read(io.BytesIO(raw), dtype='float32')
                    if audio.ndim > 1:
                        audio = np.mean(audio, axis=-1)
                    waves, rate = self.model.generate_voice_clone(**options, ref_audio=(audio, rate),
                        ref_text=voice.get('transcript'), x_vector_only_mode=voice.get('speaker_only', False))
            self._check_cancel(cancel)
            if len(waves) != 1 or rate <= 0 or not len(waves[0]):
                raise SpeechUnavailable('Qwen returned no complete waveform')
            output = io.BytesIO()
            sf.write(output, waves[0], rate, subtype='PCM_16', format='WAV')
            return SpeechResult(output.getvalue(), self.name)
        finally:
            # Cloning samples and generated arrays do not become resident state.
            audio = waves = None

    def unload(self):
        """Free model references and finish CUDA cleanup before admission release."""
        import torch
        self.model = None
        self.resident_device = None
        gc.collect()
        if self.device != 'cpu':
            with torch.cuda.device(self.device):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
