"""Whisper inference with Kadan-owned memory and no implicit downloads."""
import base64
import binascii
from functools import lru_cache
from contextlib import ExitStack
import gc
import hashlib
import io
import json
import os
import threading
import sys
import wave

import numpy as np

from api.inference.placement import select_device
from api.inference.resources import ResourceBusy, ResourceExhausted, ResourceCancelled
from api.services.model_downloads import model_manager
from api.inference.errors import InferenceFailure
from api.services.chat_runtime import chat_runtime
from api.inference.stt.catalog import get_whisper_checkpoints, checkpoint
from api.inference.failure_cleanup import clear_failure_frames


class WhisperTranscriber:
    def __init__(self, factory=None, resources=None, store=None):
        """Construct coordination only; allocate weights on an explicit request."""
        self.factory = factory
        self.resources = resources
        self.store = store or model_manager
        self.lock = threading.Lock()
        self.whisper_model = None
        self.name = None
        self.host_reservation = self.device_reservation = None
        self.device = 'cpu'

    def selected(self):
        """Restore a canonical selection without loading model weights."""
        try:
            name = checkpoint(json.loads((self.store.root / 'whisper-selection.json').read_text())['model']).name
            return name if name in get_whisper_checkpoints() else next(iter(get_whisper_checkpoints()), None)
        except (OSError, ValueError, KeyError, TypeError):
            return next(iter(get_whisper_checkpoints()), None)

    def select(self, name):
        """Persist a validated checkpoint name atomically, without downloading it."""
        name = checkpoint(name).name
        if name not in get_whisper_checkpoints():
            raise ValueError("Whisper checkpoint is not enabled in this version")
        with self.lock:
            self.store.root.mkdir(parents=True, exist_ok=True)
            temporary = self.store.root / 'whisper-selection.tmp'
            temporary.write_text(json.dumps({'model': name}))
            temporary.replace(self.store.root / 'whisper-selection.json')
        return name

    def _offload_locked(self):
        if self.device != 'cpu' and self.whisper_model is not None:
            self.whisper_model.to('cpu')
            torch = sys.modules.get('torch')
            if torch is not None and torch.cuda.is_initialized():
                with torch.cuda.device(self.device):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            self.device = 'cpu'
        if self.device_reservation is not None:
            self.device_reservation.release()
            self.device_reservation = None

    def _clear_locked(self):
        # Pressure eviction disposes weights directly. Copying CUDA weights to
        # RAM immediately before disposal would require avoidable host headroom.
        self.whisper_model = self.name = None
        gc.collect()
        torch = sys.modules.get('torch')
        if self.device != 'cpu' and torch is not None and torch.cuda.is_initialized():
            with torch.cuda.device(self.device):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        self.device = 'cpu'
        if self.device_reservation is not None:
            self.device_reservation.release()
            self.device_reservation = None
        if self.host_reservation is not None:
            self.host_reservation.release()
            self.host_reservation = None

    def _offload(self):
        if not self.lock.acquire(blocking=False):
            raise ResourceBusy('Whisper is active')
        try:
            self._offload_locked()
        finally:
            self.lock.release()

    def _evict(self):
        if not self.lock.acquire(blocking=False):
            raise ResourceBusy('Whisper is active')
        try:
            self._clear_locked()
        finally:
            self.lock.release()

    def close(self):
        with self.lock:
            self._clear_locked()

    def _load_locked(self, entry, name, resources, device, device_budget, cancel):
        if self.whisper_model is None:
            _, directory = self.store.get_checkpoint(f'whisper-{entry.name}')
            path = directory / f'{entry.name}.pt'
            if not path.is_file() or path.is_symlink():
                raise InferenceFailure('Download this Whisper checkpoint completely in Settings.', 409)
            with path.open('rb') as source:
                if hashlib.file_digest(source, 'sha256').hexdigest() != entry.sha256:
                    raise InferenceFailure('Whisper checkpoint integrity verification failed.', 409)
            if cancel.is_set():
                raise ResourceCancelled('Transcription cancelled')
            self.whisper_model = (self.factory or load_whisper)(str(path), device='cpu')
            self.name = name
        if self.device != device:
            self._offload_locked()
            if device != 'cpu':
                self.device_reservation = resources.reserve('whisper:device', 'speech',
                    device_bytes={int(device[5:]): device_budget}, evict=self._offload, cancel_event=cancel)
                try:
                    self.whisper_model.to(device)
                    self.device = device
                except BaseException:
                    self.whisper_model.to('cpu')
                    self.device_reservation.release()
                    self.device_reservation = None
                    raise

    def _select_device(self, resources, entry, requested, extra=0):
        retained = None
        if self.name == entry.name and self.device != 'cpu' and self.device_reservation is not None:
            retained = (int(self.device[5:]), entry.device_memory_gib * 1024**3)
        return select_device(resources, entry.device_memory_gib * 1024**3 + extra,
                             requested, allow_cpu=True, retained=retained)

    def load(self, model=None, cancel=None):
        """Load/promote one selected checkpoint under the existing host lease."""
        cancel = cancel or threading.Event()
        name = model or self.selected()
        entry = checkpoint(name)
        if entry.name not in get_whisper_checkpoints():
            raise InferenceFailure('Whisper checkpoint is not enabled in this version.', 422)
        with self.lock:
            resources = self.resources or chat_runtime.ensure_resources()
            device = os.environ.get('KADAN_WHISPER_DEVICE', 'auto')
            if self.name != name:
                self._clear_locked()
            device = self._select_device(resources, entry, device)
            try:
                if self.host_reservation is None:
                    self.host_reservation = resources.reserve('whisper:host', 'speech',
                        host_bytes=entry.memory_gib * 1024**3, evict=self._evict, cancel_event=cancel)
                with self.host_reservation.lease(cancel):
                    self._load_locked(entry, name, resources, device, entry.device_memory_gib * 1024**3, cancel)
            except BaseException as exc:
                clear_failure_frames(exc)
                if self.whisper_model is None:
                    self._clear_locked()
                raise
            finally:
                if self.whisper_model is None and self.host_reservation is not None:
                    self.host_reservation.release()
                    self.host_reservation = None

    def offload_to_ram(self, cancel=None):
        resources = self.resources or chat_runtime.ensure_resources()
        resources.offload_workload_devices('speech', cancel)

    def transcribe(self, audio, model=None, language=None, cancel=None):
        """Retain idle CPU weights; leased device state is independently evictable."""
        cancel = cancel or threading.Event()
        name = model or self.selected()
        if name is None:
            raise InferenceFailure('No Whisper checkpoint is enabled in this version.', 503)
        entry = checkpoint(name)
        if entry.name not in get_whisper_checkpoints():
            raise InferenceFailure('Whisper checkpoint is not enabled in this version.', 422)
        if entry.name.endswith('.en') and language not in (None, 'en'):
            raise InferenceFailure('This Whisper checkpoint supports English only.', 422)
        if not audio.startswith('data:audio/wav;base64,'):
            raise InferenceFailure('Supply a base64 PCM WAV data URL. Audio references and URLs are not fetched.', 422)
        if not self.lock.acquire(blocking=False):
            raise InferenceFailure('A transcription is already running.', 409)
        audio_reservation = None
        samples = payload = frames = None
        hooks = []

        def check_cancel(*_):
            if cancel.is_set():
                raise ResourceCancelled('Transcription cancelled')

        try:
            check_cancel()
            device = os.environ.get('KADAN_WHISPER_DEVICE', 'auto')
            resources = self.resources or chat_runtime.ensure_resources()
            budget = entry.memory_gib * 1024**3
            device_budget = entry.device_memory_gib * 1024**3
            device = self._select_device(resources, entry, device, len(audio) * 16)
            if self.name != name:
                self._clear_locked()
            if self.host_reservation is None:
                self.host_reservation = resources.reserve('whisper:host', 'speech', host_bytes=budget,
                    evict=self._evict, cancel_event=cancel)
            with ExitStack() as leases:
                leases.enter_context(self.host_reservation.lease(cancel))
                audio_budget = len(audio) * 16
                audio_reservation = resources.reserve('whisper:audio', 'speech', host_bytes=audio_budget,
                    device_bytes={} if device == 'cpu' else {int(device[5:]): audio_budget}, cancel_event=cancel)
                leases.enter_context(audio_reservation.lease(cancel))
                try:
                    payload = base64.b64decode(audio.split(',', 1)[1], validate=True)
                    with wave.open(io.BytesIO(payload), 'rb') as wav:
                        if (wav.getsampwidth(), wav.getnchannels(), wav.getframerate(), wav.getcomptype()) != (2, 1, 16000, 'NONE'):
                            raise ValueError('Expected mono 16-bit PCM WAV at 16000 Hz')
                        frames = wav.readframes(wav.getnframes())
                        if not frames or len(frames) != wav.getnframes() * 2:
                            raise ValueError('WAV contains no audio or is truncated')
                        samples = np.frombuffer(frames, dtype='<i2').astype(np.float32) / 32768.0
                except (ValueError, EOFError, wave.Error, binascii.Error) as exc:
                    raise InferenceFailure(f'Invalid audio: {exc}', 422) from exc
                self._load_locked(entry, name, resources, device, device_budget, cancel)
                if self.device_reservation is not None:
                    leases.enter_context(self.device_reservation.lease(cancel))
                # Whisper has no cancellation argument. Module boundaries
                # provide cooperative interruption without freeing live tensors.
                if hasattr(self.whisper_model, 'modules'):
                    hooks = [module.register_forward_pre_hook(check_cancel) for module in self.whisper_model.modules()]
                check_cancel()
                result = self.whisper_model.transcribe(samples, language='en' if entry.name.endswith('.en') else language,
                    task='transcribe', fp16=device != 'cpu', verbose=None)
                check_cancel()
                if not isinstance(result, dict) or not isinstance(result.get('text'), str):
                    raise InferenceFailure('Whisper returned an invalid transcript.', 502)
                return {'text': result['text'], 'raw_text': result['text'],
                        'language': result.get('language', language or 'en'), 'model': entry.name,
                        'formatting_status': 'disabled'}
        except Exception as exc:
            clear_failure_frames(exc)
            if not isinstance(exc, (InferenceFailure, ResourceCancelled)):
                self._clear_locked()
            if isinstance(exc, (InferenceFailure, ResourceCancelled)):
                raise
            status = 503 if isinstance(exc, (ResourceBusy, ResourceExhausted, ValueError, OSError)) else 502
            raise InferenceFailure(f'Transcription failed: {exc}', status) from exc
        finally:
            for hook in hooks:
                hook.remove()
            samples = payload = frames = None
            gc.collect()
            torch = sys.modules.get('torch')
            if self.device != 'cpu' and torch is not None and torch.cuda.is_initialized():
                with torch.cuda.device(self.device):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            if audio_reservation is not None:
                audio_reservation.release()
            if self.whisper_model is None and self.host_reservation is not None:
                self.host_reservation.release()
                self.host_reservation = None
            self.lock.release()


def load_whisper(path, device):
    """Import the optional native package only at inference; never download implicitly."""
    # Optional native imports stay lazy so missing inference extras do not stop API startup.
    try:
        import whisper
    except ImportError as exc:
        raise InferenceFailure('Install the pinned Whisper inference dependencies before transcription.') from exc
    return whisper.load_model(path, device=device)


_manager_lock = threading.Lock()


@lru_cache(maxsize=1)
def _cached_whisper_transcriber():
    return WhisperTranscriber()


def get_whisper_transcriber():
    """Lazily share one coordinator, including concurrent first requests.

    Hold the lock outside lru_cache so the first result is cached before another
    caller can construct a manager with a separate admission lock.
    """
    with _manager_lock:
        return _cached_whisper_transcriber()
