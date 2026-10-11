"""Persist Whisper checkpoint selection without owning inference."""
import json
import threading

from api.services.model_downloads import model_manager
from api.inference.stt.catalog import get_whisper_checkpoints, checkpoint


class WhisperSelection:
    def __init__(self, store=None):
        self.store = store or model_manager
        self.lock = threading.Lock()

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


whisper_selection = WhisperSelection()
