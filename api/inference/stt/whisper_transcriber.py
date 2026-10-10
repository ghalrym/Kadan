"""Whisper selection and native-worker coordination; no Python model execution."""
from functools import lru_cache
import json
import threading

from api.services.model_downloads import model_manager
from api.inference.errors import InferenceFailure
from api.inference.stt.native_worker import NativeWhisper
from api.inference.stt.catalog import get_whisper_checkpoints, checkpoint


class WhisperTranscriber:
    def __init__(self, *, resources=None, store=None, worker=None):
        """Construct coordination only; allocate weights on an admitted request."""
        self._cpp = worker if worker is not None else NativeWhisper(resources)
        self.store = store or model_manager
        self.lock = threading.Lock()
        self.whisper_model = None

    def _native_model(self, model):
        entry = checkpoint(model or self.selected())
        if entry.name not in get_whisper_checkpoints():
            raise InferenceFailure('Whisper checkpoint is not enabled in this version.', 422)
        return entry.name

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

    def close(self):
        self._cpp.close()
        self.whisper_model = None

    def load(self, model=None, cancel=None):
        self._cpp.load(self._native_model(model), cancel)
        self.whisper_model = self._cpp

    def offload_to_ram(self, cancel=None):
        return self._cpp.offload_to_ram(cancel)

    def transcribe(self, audio, model=None, language=None, cancel=None):
        result = self._cpp.transcribe(audio, self._native_model(model), language, cancel)
        self.whisper_model = self._cpp
        return result


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
