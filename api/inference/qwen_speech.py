"""Qwen-specific validation, checkpoint preparation and resident worker adapter."""
import base64
import binascii
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import threading
import time

from api.inference.resources import ResourceCancelled
from api.inference.speech import SpeechInput, SpeechModel, SpeechPlan, SpeechResult, SpeechUnavailable
from api.inference.speech_process import OwnedSpeechProcess
from api.services.model_downloads import model_manager
from api.services.qwen_tts_catalog import SPEECH_MODELS, SPEAKERS, LANGUAGES
from api.services import speech_enabled


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
        python = os.environ.get('KADAN_QWEN_TTS_PYTHON')
        if not python or not Path(python).is_file():
            raise SpeechUnavailable('Configure the isolated Qwen3-TTS worker environment before generating speech.')
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
        return SpeechPlan(('qwen', model.id, model.revision, str(checkpoint), python, device),
            budget, lambda: QwenSpeechSession(python, checkpoint, device, model.name),
            {} if device == 'cpu' else {int(device[5:]): budget})


class QwenSpeechSession:
    """One loaded model per owned process, reused until eviction/unload/failure."""
    def __init__(self, python, checkpoint, device, name):
        self.python, self.checkpoint, self.device, self.name = python, checkpoint, device, name
        self.worker = OwnedSpeechProcess()
        self.directory = None
        self.sequence = 0

    def _wait(self, path: Path, cancel: threading.Event):
        deadline = time.monotonic() + 1800
        while True:
            if cancel.is_set():
                raise ResourceCancelled('Speech generation cancelled')
            if self.worker.poll() is not None:
                raise SpeechUnavailable('Qwen3-TTS worker exited; check its environment and local checkpoint.')
            if path.exists():
                status = json.loads(path.read_text())
                if status.get('ok') is not True:
                    raise SpeechUnavailable('Qwen3-TTS worker failed; its residency will be unloaded.')
                return
            if time.monotonic() >= deadline:
                raise SpeechUnavailable('Qwen3-TTS worker timed out.')
            cancel.wait(.05)

    def load(self, cancel):
        self.directory = tempfile.TemporaryDirectory(prefix='kadan-speech-resident-')
        root = Path(self.directory.name)
        (root / 'config.json').write_text(json.dumps(dict(checkpoint=str(self.checkpoint), device=self.device)))
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
        worker = Path(__file__).parents[1] / 'workers' / 'kadan_qwen_tts_worker.py'
        with (root / 'worker.log').open('wb') as log:
            self.worker.start([self.python, str(worker), '--serve', str(root)], env=env, log=log)
        self._wait(root / 'ready.json', cancel)

    def generate(self, request, cancel):
        self.sequence += 1
        root = Path(self.directory.name)
        directory = tempfile.mkdtemp(prefix='request-', dir=root)
        job = Path(directory)
        try:
            payload = root / f'{self.sequence}.pending'
            payload.write_text(json.dumps(dict(request=asdict(request), output=str(job / 'output.wav'),
                                               result=str(job / 'result.json'))))
            payload.replace(root / f'{self.sequence}.request.json')
            self._wait(job / 'result.json', cancel)
            result = SpeechResult((job / 'output.wav').read_bytes(), self.name)
        except BaseException:
            # The runtime unloads/reaps the worker before cleaning its root.
            raise
        else:
            for path in job.iterdir():
                path.unlink()
            job.rmdir()
            return result

    def unload(self):
        self.worker.stop()
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None
