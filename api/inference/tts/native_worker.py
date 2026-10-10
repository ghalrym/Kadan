"""Retained native CPU TTS session; no Python model execution."""
import os
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from api.inference.line_protocol import LineProtocolError, LineProtocolProcess
from api.inference.resources import ResourceCancelled
from api.inference.tts.speech_runtime import SpeechResult, SpeechUnavailable

HOST_BUDGET = 24 * 1024**3
PROCESS_BUDGET = HOST_BUDGET + 256 * 1024**2
MODEL = 'qwen-tts-1.7b-custom'


def validate(request):
    if (request.model_id != MODEL or request.language != 'English'
            or request.voice.get('mode') != 'custom' or request.voice.get('speaker') != 'Ryan'
            or request.voice.get('instruction')):
        raise SpeechUnavailable('Native TTS currently supports the 1.7B CustomVoice checkpoint, English and Ryan without instruction control.')
    if len(request.script.encode('utf-8')) > 32000:
        raise SpeechUnavailable('Native TTS requires 1..32000 UTF-8 bytes of text.')


def check_cancel(cancel):
    if cancel.is_set():
        raise ResourceCancelled('Speech generation cancelled')


class NativeSpeechSession:
    """SpeechRuntime owns the lifetime reservation and serializes this session.

    Process termination/reaping finishes before unload returns. A failed reap
    propagates, leaving SpeechRuntime's admission reserved for a retry.
    """
    def __init__(self, checkpoint, tokenizer, binary, name, process_factory=LineProtocolProcess):
        self.checkpoint, self.tokenizer, self.binary, self.name = checkpoint, tokenizer, binary, name
        self.process_factory = process_factory
        self.child = self.workspace = self.baseline = None
        self.quarantined = False

    def check_execution_state(self):
        if self.quarantined:
            raise SpeechUnavailable("Native TTS child cleanup is unconfirmed; retry unload before another job.")

    def load(self, cancel):
        self.check_execution_state()
        check_cancel(cancel)
        if self.child is not None:
            raise SpeechUnavailable('Native speech session is already loaded.')
        self.workspace = Path(tempfile.mkdtemp(prefix='kadan-tts-'))
        self.child = self.process_factory()
        self.child.start([str(self.binary), str(self.checkpoint), str(self.tokenizer), str(self.workspace)],
                         env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
        try:
            ready = self.child.read(600, cancel).split()
        except InterruptedError as error:
            raise ResourceCancelled('Speech loading cancelled') from error
        if len(ready) != 3 or ready[:2] != ['ready', '1'] or not ready[2].isdigit() or not 0 < int(ready[2]) <= HOST_BUDGET:
            raise LineProtocolError('Invalid native TTS ready response')
        self.baseline = int(ready[2])

    def restore(self, cancel):
        check_cancel(cancel)
        if self.child is None or self.baseline is None:
            raise SpeechUnavailable('Native speech session is not loaded.')

    def offload_to_ram(self):
        pass  # CPU resident; host pressure invokes unload through SpeechRuntime.

    def generate(self, request, cancel):
        validate(request)
        if not request.script:
            raise SpeechUnavailable("Native TTS requires nonempty text.")
        self.restore(cancel)
        (self.workspace / 'text.txt').write_text(request.script, encoding='utf-8')
        output = self.workspace / 'audio.wav'
        if output.exists():
            output.unlink()
        try:
            reply = self.child.exchange('speak', 7200, cancel).split()
        except InterruptedError as error:
            raise ResourceCancelled('Speech generation cancelled') from error
        if (len(reply) != 4 or reply[0] != 'done' or not all(v.isdigit() for v in reply[1:])
                or not 1 <= int(reply[2]) <= 300 or int(reply[1]) != 44 + int(reply[2]) * 1920 * 2
                or int(reply[3]) != self.baseline):
            raise LineProtocolError('Invalid native TTS completion response')
        if output.is_symlink() or not output.is_file() or output.stat().st_size != int(reply[1]):
            raise LineProtocolError('Invalid native TTS waveform artifact')
        data = output.read_bytes()
        check_cancel(cancel)
        return SpeechResult(data, self.name)

    def unload(self):
        if self.child is not None:
            try:
                self.child.stop()
            except BaseException:
                self.quarantined = True
                raise
            self.child = None
        if self.workspace is not None:
            shutil.rmtree(self.workspace)
            self.workspace = None
        self.baseline = None
        self.quarantined = False


def verify_tokenizer(checkpoint, tokenizer):
    manifest = tokenizer.parent / 'manifest.json'
    if manifest.stat().st_size > 16384:
        raise SpeechUnavailable('Invalid native TTS tokenizer manifest.')
    metadata = json.loads(manifest.read_text())
    if metadata.get('version') != 1:
        raise SpeechUnavailable('Unsupported native TTS tokenizer manifest.')
    expected = {'vocab.json', 'merges.txt', 'tokenizer_config.json', 'config.json'}
    if set(metadata.get('source_sha256', {})) != expected:
        raise SpeechUnavailable('Incomplete native TTS tokenizer manifest.')
    for path, digest in [(tokenizer, metadata['tokenizer_sha256']),
                         *((Path(checkpoint) / name, metadata['source_sha256'][name]) for name in sorted(expected))]:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 * 1024**2:
            raise SpeechUnavailable('Invalid native TTS tokenizer asset.')
        with path.open('rb') as source:
            if hashlib.file_digest(source, 'sha256').hexdigest() != digest:
                raise SpeechUnavailable('Native TTS tokenizer does not match the selected checkpoint.')
