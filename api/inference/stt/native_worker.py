"""Owned C++ Whisper CPU process; Python handles WAV transport and admission only."""
import base64
import binascii
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import threading
import wave

import numpy as np

from api.inference.errors import InferenceFailure
from api.inference.failure_cleanup import clear_failure_frames
from api.inference.line_protocol import LineProtocolError, LineProtocolProcess
from api.inference.resources import ResourceBusy, ResourceCancelled
from api.inference.stt.catalog import checkpoint
from api.services.chat_runtime import chat_runtime

HOST_BUDGET = 16 * 1024**3
PROCESS_BUDGET = HOST_BUDGET + 256 * 1024**2
MAX_PCM = 480000


def cancelled(event):
    if event is not None and event.is_set():
        raise ResourceCancelled('Transcription cancelled')


def resolve(model):
    entry = checkpoint(model)
    binary = Path(os.environ['KADAN_NATIVE_WHISPER_WORKER'])
    root = Path(os.environ.get('KADAN_NATIVE_WHISPER_MODEL_ROOT', ''))
    assets = Path(os.environ.get('KADAN_NATIVE_WHISPER_ASSET_ROOT', ''))
    if not binary.is_absolute() or not root.is_absolute() or not assets.is_absolute():
        raise InferenceFailure('Configure absolute native Whisper worker, model and asset paths.')
    capability = subprocess.run([str(binary), '--capabilities'], capture_output=True, timeout=10, check=True)
    if capability.stdout != f'whisper 1 cpu en {MAX_PCM} {HOST_BUDGET}\n'.encode():
        raise InferenceFailure('Unsupported native Whisper worker capabilities.')
    report_path = root / 'export.json'
    if report_path.stat().st_size > 16384:
        raise InferenceFailure('Invalid Whisper export manifest.')
    report = json.loads(report_path.read_text())
    weight = root / 'model.safetensors'
    if report['source_sha256'] != entry.sha256 or weight.stat().st_size != report['output_bytes']:
        raise InferenceFailure('Native Whisper export does not match the selected checkpoint.')
    with weight.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != report['output_sha256']:
            raise InferenceFailure('Native Whisper export integrity verification failed.')
    assets_report = assets / 'assets.json'
    if assets_report.stat().st_size > 16384:
        raise InferenceFailure('Invalid Whisper asset manifest.')
    metadata = json.loads(assets_report.read_text())
    if metadata['language'] != 'en' or metadata['whisper_version'] != '20250625' or metadata['vocabulary'] != report['dimensions']['n_vocab']:
        raise InferenceFailure('Whisper decoding assets do not match the checkpoint.')
    for name, expected in [('english.tokens', metadata['token_sha256']),
                           (f"mel-{report['dimensions']['n_mels']}.f32", metadata['mel_sha256'][str(report['dimensions']['n_mels'])])]:
        path = assets / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024**2:
            raise InferenceFailure('Invalid native Whisper asset file.')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise InferenceFailure('Native Whisper asset integrity verification failed.')
    return [str(binary), str(root), weight.name, str(root / 'dimensions.txt'), str(assets)]


def pcm(audio):
    if not isinstance(audio, str) or not audio.startswith('data:audio/wav;base64,') or len(audio) > 2 * 1024**2:
        raise InferenceFailure('Supply at most 30 seconds of mono 16-kHz PCM WAV.', 422)
    try:
        raw = base64.b64decode(audio.split(',', 1)[1], validate=True)
        with wave.open(io.BytesIO(raw), 'rb') as source:
            count = source.getnframes()
            if (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype()) != (1, 2, 16000, 'NONE') or not 0 < count <= MAX_PCM:
                raise ValueError('Expected 1..480000 mono 16-bit samples at 16000 Hz')
            frames = source.readframes(count)
            if len(frames) != count * 2:
                raise ValueError('Truncated WAV')
        return (np.frombuffer(frames, dtype='<i2').astype('<f4') / 32768).tobytes()
    except (ValueError, EOFError, wave.Error, binascii.Error) as error:
        raise InferenceFailure(f'Invalid audio: {error}', 422) from error


class NativeWhisper:
    def __init__(self, resources=None, resolver=resolve, process_factory=LineProtocolProcess):
        self.resources, self.resolver, self.process_factory = resources, resolver, process_factory
        self.lock = threading.Lock()
        self.child = self.admission = self.workspace = self.model = None
        self.baseline = None
        self.quarantined = False
        self.owner = f'whisper-native:{id(self)}'

    def check_execution_state(self):
        if self.quarantined:
            raise InferenceFailure('Whisper child cleanup is unconfirmed; close the runtime before another job.')

    def _close(self):
        try:
            if self.child is not None:
                self.child.stop()
                self.child = None
        except BaseException:
            self.quarantined = True
            raise
        if self.admission is not None:
            self.admission.release()
            self.admission = None
        if self.workspace is not None:
            shutil.rmtree(self.workspace)
            self.workspace = None
        self.model = self.baseline = None
        self.quarantined = False

    def close(self):
        with self.lock:
            self._close()

    def evict(self):
        if not self.lock.acquire(blocking=False):
            raise ResourceBusy('Whisper is executing')
        try:
            self._close()
        finally:
            self.lock.release()

    def offload_to_ram(self, cancel=None):
        cancelled(cancel)  # CPU weights remain resident until host pressure/close.

    def _load(self, model, cancel):
        cancelled(cancel)
        self.check_execution_state()
        if self.model == model and self.child is not None:
            return
        command = self.resolver(model)
        self._close()
        resources = self.resources or chat_runtime.ensure_resources()
        self.admission = resources.reserve(self.owner, 'speech', host_bytes=PROCESS_BUDGET, evict=self.evict, cancel_event=cancel)
        with self.admission.lease(cancel):
            self.workspace = Path(tempfile.mkdtemp(prefix='kadan-whisper-'))
            self.child = self.process_factory()
            self.child.start([*command, str(self.workspace)], env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
            ready = self.child.read(600, cancel).split()
            if len(ready) != 3 or ready[:2] != ['ready', '1'] or not ready[2].isdigit() or not 0 < int(ready[2]) <= HOST_BUDGET:
                raise LineProtocolError('Invalid native Whisper ready response')
            self.baseline = int(ready[2])
            self.model = model

    def load(self, model, cancel=None):
        if not self.lock.acquire(blocking=False):
            raise ResourceBusy('Whisper is executing')
        try:
            self._load(model, cancel)
        except BaseException:
            self._close()
            raise
        finally:
            self.lock.release()

    def transcribe(self, audio, model, language=None, cancel=None):
        if language not in (None, 'en'):
            raise InferenceFailure('The native Whisper worker currently supports English only.', 422)
        if not self.lock.acquire(blocking=False):
            raise ResourceBusy('Whisper is executing')
        try:
            cancelled(cancel)
            resources = self.resources or chat_runtime.ensure_resources()
            # Includes base64, WAV, NumPy conversion and IPC publication buffers.
            transient = resources.reserve(self.owner + ':audio', 'speech', host_bytes=16 * 1024**2, cancel_event=cancel)
            try:
                with transient.lease(cancel):
                    samples = pcm(audio)
                    self._load(model, cancel)
                    with self.admission.lease(cancel):
                        (self.workspace / 'pcm.f32').write_bytes(samples)
                        output = self.workspace / 'text.txt'
                        if output.exists():
                            output.unlink()
                        reply = self.child.exchange('transcribe', 3600, cancel).split()
                        if (len(reply) != 4 or reply[0] != 'done' or not all(n.isdigit() for n in reply[1:])
                                or not 0 <= int(reply[1]) <= 172032 or not 1 <= int(reply[2]) <= 224 or int(reply[3]) != self.baseline):
                            raise LineProtocolError('Invalid native Whisper completion response')
                        if output.is_symlink() or not output.is_file() or output.stat().st_size != int(reply[1]):
                            raise LineProtocolError('Invalid native Whisper transcript artifact')
                        text = output.read_text(encoding='utf-8')
                        cancelled(cancel)
                        return dict(text=text, raw_text=text, language='en', model=model, formatting_status='disabled')
            except BaseException as error:
                clear_failure_frames(error)
                raise
            finally:
                samples = None
                transient.release()
        except BaseException as error:
            self._close()
            if isinstance(error, InterruptedError):
                raise ResourceCancelled('Transcription cancelled') from error
            raise
        finally:
            self.lock.release()
