"""Native Whisper inference with Kadan-owned memory and no implicit downloads."""
import base64
import binascii
import gc
import hashlib
import io
import json
import os
import threading
import sys
import wave

import numpy as np

from api.inference.resources import ResourceBusy, ResourceExhausted
from api.services.model_downloads import model_manager
from api.services.runtime import RuntimeFailure, runtime_manager
from api.services.whisper_catalog import CHECKPOINTS, checkpoint
from api.services.decisions import clear_failure_frames


class TranscriptionManager:
    def __init__(self, factory=None, resources=None, store=None):
        """Construct coordination only; allocate weights on an explicit request."""
        self.factory = factory
        self.resources = resources
        self.store = store or model_manager
        self.lock = threading.Lock()

    def selected(self):
        """Restore a canonical selection without loading model weights."""
        try:
            name = checkpoint(json.loads((self.store.root / 'whisper-selection.json').read_text())['model']).name
            return name if name in CHECKPOINTS else next(iter(CHECKPOINTS), None)
        except (OSError, ValueError, KeyError, TypeError):
            return next(iter(CHECKPOINTS), None)

    def select(self, name):
        """Persist a validated checkpoint name atomically, without downloading it."""
        name = checkpoint(name).name
        if name not in CHECKPOINTS:
            raise ValueError("Whisper checkpoint is not enabled in this version")
        with self.lock:
            self.store.root.mkdir(parents=True, exist_ok=True)
            temporary = self.store.root / 'whisper-selection.tmp'
            temporary.write_text(json.dumps({'model': name}))
            temporary.replace(self.store.root / 'whisper-selection.json')
        return name

    def transcribe(self, audio, model=None, language=None):
        """Decode PCM WAV and run one native request; ownership lasts through cleanup."""
        name = model or self.selected()
        if name is None:
            raise RuntimeFailure("No Whisper checkpoint is enabled in this version.", 503)
        entry = checkpoint(name)
        if entry.name not in CHECKPOINTS:
            raise RuntimeFailure("Whisper checkpoint is not enabled in this version.", 422)
        if entry.name.endswith('.en') and language not in (None, 'en'):
            raise RuntimeFailure('This Whisper checkpoint supports English only.', 422)
        if not audio.startswith('data:audio/wav;base64,'):
            raise RuntimeFailure('Supply a base64 PCM WAV data URL. Audio references and URLs are not fetched.', 422)
        if not self.lock.acquire(blocking=False):
            raise RuntimeFailure('A transcription is already running.', 409)
        reservation = None
        native = samples = payload = frames = None
        try:
            # Resolve only Kadan's completed store; native Whisper never sees a model alias.
            _, directory = self.store.get_checkpoint(f'whisper-{entry.name}')
            path = directory / f'{entry.name}.pt'
            if not path.is_file() or path.is_symlink():
                raise RuntimeFailure('Download this Whisper checkpoint completely in Settings.', 409)
            with path.open('rb') as source:
                if hashlib.file_digest(source, 'sha256').hexdigest() != entry.sha256:
                    raise RuntimeFailure('Whisper checkpoint integrity verification failed.', 409)
            device = os.environ.get('KADAN_WHISPER_DEVICE', 'cpu')
            if device != 'cpu' and not (device.startswith('cuda:') and device[5:].isdecimal()):
                raise RuntimeFailure('KADAN_WHISPER_DEVICE must be cpu or cuda:<index>.')
            resources = self.resources or runtime_manager.ensure_resources()
            # Include encoded/decoded audio and full-clip mel workspace before decoding.
            audio_budget = len(audio) * 16
            budget = entry.memory_gib * 1024**3
            reservation = resources.reserve('whisper', 'speech', host_bytes=budget + audio_budget,
                device_bytes={} if device == 'cpu' else {int(device[5:]): budget + audio_budget})
            with reservation.lease():
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
                    raise RuntimeFailure(f'Invalid audio: {exc}', 422) from exc
                factory = self.factory or load_whisper
                native = factory(str(path), device=device)
                result = native.transcribe(samples, language='en' if entry.name.endswith('.en') else language,
                    task='transcribe', fp16=device != 'cpu', verbose=None)
                if not isinstance(result, dict) or not isinstance(result.get('text'), str):
                    raise RuntimeFailure('Whisper returned an invalid transcript.', 502)
                return {'text': result['text'], 'raw_text': result['text'],
                        'language': result.get('language', language or 'en'), 'model': entry.name,
                        'formatting_status': 'disabled'}
        except Exception as exc:
            clear_failure_frames(exc)
            if isinstance(exc, RuntimeFailure):
                raise
            status = 503 if isinstance(exc, (ResourceBusy, ResourceExhausted, ValueError, OSError)) else 502
            raise RuntimeFailure(f'Transcription failed: {exc}', status) from exc
        finally:
            # Release real allocations before returning their accounting budget.
            native = samples = payload = frames = None
            gc.collect()
            torch = sys.modules.get('torch')
            if torch is not None and torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            if reservation is not None:
                reservation.release()
            self.lock.release()


def load_whisper(path, device):
    """Import the optional native package only at inference; never download implicitly."""
    # Optional native imports stay lazy so missing inference extras do not stop API startup.
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeFailure('Install the pinned Whisper inference dependencies before transcription.') from exc
    return whisper.load_model(path, device=device)


transcription_manager = TranscriptionManager()
