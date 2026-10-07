"""Qwen-specific validation, checkpoint preparation and in-process resident adapter."""
import base64
import binascii
import gc
import io
import logging
import os
import threading

from api.inference.resources import ResourceCancelled
from api.inference.tts.runtime import SpeechInput, SpeechModel, SpeechPlan, SpeechResult, SpeechUnavailable
from api.services.model_downloads import model_manager
from api.inference.tts.catalog import SPEECH_MODELS, SPEAKERS, LANGUAGES
from api.inference.tts import enabled as speech_enabled


log = logging.getLogger(__name__)


class QwenSpeechProvider:
    def models(self):
        return tuple(SpeechModel(item.id, item.name, item.mode,
            SPEAKERS if item.mode == 'custom' else (), item.id == 'qwen-tts-1.7b-custom',
            'Ryan' if item.mode == 'custom' else None)
            for item in SPEECH_MODELS.values())

    def enabled(self, model_id):
        return model_id in speech_enabled.ENABLED_SPEECH_MODELS

    def validate(self, request: SpeechInput):
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

    def prepare(self, request):
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
        device = os.environ.get('KADAN_QWEN_TTS_DEVICE', 'cuda:0')
        if device != 'cpu' and not (device.startswith('cuda:') and device[5:].isdigit()):
            raise SpeechUnavailable('Qwen speech device must be cpu or cuda:N.')
        budget = model.estimated_bytes * 3 + 2 * 1024**3
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
