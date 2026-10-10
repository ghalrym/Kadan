"""Native H3 process ownership under the existing global request consumer.

C++ owns tokenization, model math, profile geometry and CUDA allocations. Python
owns admission, cancellation, the codec child and atomic output publication.
No SGLang or Torch model fallback is available in this provider.
"""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import tempfile
import threading
import time

from api.inference.errors import InferenceFailure
from api.inference.line_protocol import LineProtocolError, LineProtocolProcess
from api.inference.resources import ResourceBusy, ResourceCancelled
from api.services.chat_runtime import chat_runtime
from api.services.model_catalog import H3_INT8_REVISION
from api.services.model_downloads import model_manager

GIB = 1024**3
HOST_BUDGET = 128 * GIB
DEVICE_BUDGET = 2 * GIB
CONTEXT_BUDGET = GIB


def check_cancel(event):
    if event is not None and event.is_set():
        raise ResourceCancelled('Video generation cancelled')


def decode_response(frame):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise LineProtocolError('Duplicate H3 response field')
            result[key] = value
        return result
    def constant(_):
        raise LineProtocolError('Nonfinite H3 response')
    try:
        return json.loads(frame, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise LineProtocolError('Invalid H3 JSON') from error


class H3Process(LineProtocolProcess):
    def exchange(self, request, timeout, cancel=None):
        check_cancel(cancel)
        if self.buffer or self.process.poll() is not None:
            raise LineProtocolError('H3 worker unavailable or unsolicited output')
        frame = json.dumps(request, ensure_ascii=False, separators=(',', ':')).encode() + b'\n'
        if len(frame) > 65536:
            raise InferenceFailure('H3 request exceeds its transport byte limit.', 422)
        offset = 0
        deadline = time.monotonic() + timeout
        os.set_blocking(self.process.stdin.fileno(), False)
        self.selector.register(self.process.stdin, selectors.EVENT_WRITE, 'in')
        while True:
            check_cancel(cancel)
            if time.monotonic() >= deadline:
                raise TimeoutError('H3 worker deadline expired')
            for key, _ in self.selector.select(.05):
                try:
                    if key.data == 'in':
                        offset += os.write(key.fileobj.fileno(), frame[offset:offset + 4096])
                        if offset == len(frame):
                            self.selector.unregister(key.fileobj)
                        continue
                    data = os.read(key.fileobj.fileno(), 4096)
                except BlockingIOError:
                    continue
                if not data:
                    self.selector.unregister(key.fileobj)
                    if key.data == 'out':
                        raise LineProtocolError('H3 worker exited without a response')
                elif key.data == 'err':
                    self.diagnostics.extend(data)
                    del self.diagnostics[:-8192]
                else:
                    self.buffer.extend(data)
                    if len(self.buffer) > 8192:
                        raise LineProtocolError('H3 response exceeds its byte limit')
            if b'\n' in self.buffer:
                line, _, extra = self.buffer.partition(b'\n')
                if extra or offset != len(frame):
                    raise LineProtocolError('Unexpected H3 response ordering')
                self.buffer.clear()
                return decode_response(line)


def resolve_command(model_id):
    binary = Path(os.getenv('KADAN_NATIVE_H3_WORKER', '/opt/kadan/bin/kadan-h3-worker')).expanduser()
    if not binary.is_absolute() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise InferenceFailure('Native H3 worker is unavailable; configure KADAN_NATIVE_H3_WORKER.')
    # This mode only reports compile-time capabilities: no model or CUDA context.
    result = subprocess.run([str(binary), '--capabilities'], capture_output=True, timeout=10, check=True)
    expected = dict(protocol=1, cuda=True, audio=False, host_budget=HOST_BUDGET, device_budget=DEVICE_BUDGET)
    if decode_response(result.stdout) != expected:
        raise InferenceFailure('H3 requires the CUDA native worker with protocol 1.')
    entry, checkpoint = model_manager.get_checkpoint(model_id)
    if entry.revision != H3_INT8_REVISION:
        raise InferenceFailure('H3 checkpoint revision does not match the native worker.')
    names = ('FL2VA/tokenizer/tokenizer.json', 'FL2VA/text_encoder/model.safetensors',
             'FL2VA/transformer/model.safetensors',
             'loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors',
             'FL2VA/video_vae/model.safetensors')
    paths = [checkpoint / name for name in names]
    if not all(path.is_file() for path in paths):
        raise InferenceFailure('H3 checkpoint is incomplete.')
    if shutil.which('ffmpeg') is None:
        raise InferenceFailure('H3 requires the FFmpeg codec executable.')
    return [str(binary), *(str(path) for path in paths)]


def validate_artifact(response, raw, spec):
    fields = {'output', 'width', 'height', 'frames', 'audio', 'resident_bytes', 'device_resident_bytes'}
    if not isinstance(response, dict) or set(response) != fields:
        raise LineProtocolError('H3 generation failed: ' + str(response)[:200])
    if (response['output'] != str(raw) or response['audio'] is not False
            or type(response['resident_bytes']) is not int or response['resident_bytes'] != 0
            or response['device_resident_bytes'] != [0, 0]
            or any(type(n) is not int or n != 0 for n in response['device_resident_bytes'])):
        raise LineProtocolError('H3 output ownership or memory accounting mismatch')
    width, height, frames = (response[key] for key in ('width', 'height', 'frames'))
    # Native resolves profiles; verify the reported shape against the API contract.
    short = int(spec.resolution[:-1])
    expected = {'480p': {'16:9': (864, 480), '9:16': (480, 864), '1:1': (480, 480)},
                '768p': {'16:9': (1344, 768), '9:16': (768, 1344), '1:1': (768, 768)}}
    expected_frames = ((spec.duration * 24 - 5 + 16) // 17) * 17 + 5
    if (any(type(n) is not int for n in (width, height, frames))
            or (width, height) != expected[f'{short}p'][spec.aspect] or frames != expected_frames):
        raise LineProtocolError('H3 returned a different output profile')
    if raw.is_symlink() or not raw.is_file():
        raise LineProtocolError('H3 returned no regular video artifact')
    with raw.open('rb') as stream:
        header = stream.readline(256)
    expected_header = f'YUV4MPEG2 W{width} H{height} F24:1 Ip A1:1 C444 XCOLORRANGE=FULL\n'.encode()
    if header != expected_header or raw.stat().st_size != len(header) + frames * (6 + width * height * 3):
        raise LineProtocolError('H3 artifact is truncated or has an invalid header')
    return frames


class H3Provider:
    def __init__(self, model_id='h3-fl2va-int8-turbo', *, resources=None,
                 resolve=resolve_command, process_factory=H3Process):
        self.model_id, self.resources = model_id, resources
        self.resolve, self.process_factory = resolve, process_factory
        self._lock = threading.Lock()
        self._worker = self._codec = self._context = self._execution = self._workspace = None
        self._quarantined = False
        self._devices = None
        self._owner = f'h3-native:{id(self)}'

    def validate(self, spec):
        if self.model_id != 'h3-fl2va-int8-turbo':
            raise ValueError('Native H3 supports FL2VA text-to-video.')
        if spec.fps != 24 or type(spec.duration) is not int or not 4 <= spec.duration <= 15:
            raise ValueError('H3 requires 24 fps and an integer duration from 4 to 15 seconds.')
        if spec.resolution not in ('480p', '768p') or spec.aspect not in ('16:9', '9:16', '1:1'):
            raise ValueError('H3 requires 480p or 768p and a supported aspect ratio.')
        if spec.negative_prompt.strip():
            raise ValueError('The distilled H3 checkpoint does not support negative prompts.')
        if not spec.prompt or len(spec.prompt) > 8000 or '\0' in spec.prompt:
            raise ValueError('H3 requires a nonempty prompt of at most 8000 characters.')
        if type(spec.seed) is not int or not 0 <= spec.seed <= 2**64 - 1:
            raise ValueError('H3 seed must be an unsigned 64-bit integer.')

    def check_execution_state(self):
        if self._quarantined:
            raise InferenceFailure('H3 child cleanup is unconfirmed; close the runtime before another job.')

    def _close_locked(self):
        try:
            for field in ('_codec', '_worker'):
                child = getattr(self, field)
                if child is not None:
                    child.stop()
                    setattr(self, field, None)
        except BaseException:
            self._quarantined = True
            raise
        for field in ('_execution', '_context'):
            reservation = getattr(self, field)
            if reservation is not None:
                reservation.release()
                setattr(self, field, None)
        if self._workspace is not None:
            shutil.rmtree(self._workspace)
            self._workspace = None
        self._quarantined = False
        self._devices = None

    def close(self):
        with self._lock:
            self._close_locked()

    def _evict(self):
        if not self._lock.acquire(blocking=False):
            raise ResourceBusy('H3 is executing')
        try:
            self._close_locked()
        finally:
            self._lock.release()

    def offload_to_ram(self, cancellation=None):
        check_cancel(cancellation)
        self.close()  # No idle tensor bank: reap both CUDA contexts on handoff.

    def load(self, cancellation):
        self._use(None, None, cancellation)

    def generate(self, spec, output_path, cancellation):
        self.validate(spec)
        self._use(spec, Path(output_path), cancellation)

    def _encode(self, raw, target, frames, cancel):
        self._codec = LineProtocolProcess()
        self._codec.start([shutil.which('ffmpeg'), '-nostdin', '-v', 'error', '-n',
            '-threads', '1', '-i', str(raw), '-an', '-c:v', 'libx264', '-threads', '1',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-progress', 'pipe:1',
            '-nostats', '-fs', str(raw.stat().st_size + 16 * 1024**2), str(target)])
        deadline = time.monotonic() + 3600
        count, complete = 0, False
        while not complete:
            check_cancel(cancel)
            line = self._codec.read(max(.001, deadline - time.monotonic()), cancel)
            if line.startswith('frame='):
                count = int(line[6:])
            if line == 'progress=end':
                complete = True
        self._codec.finish()
        self._codec.stop()
        self._codec = None
        if count != frames or target.is_symlink() or not target.is_file() or not target.stat().st_size:
            raise LineProtocolError('H3 codec output is incomplete')

    def _use(self, spec, output, cancel):
        if not self._lock.acquire(blocking=False):
            raise ResourceBusy('H3 is already executing')
        try:
            self.check_execution_state()
            check_cancel(cancel)
            command = self.resolve(self.model_id)  # Preflight before parking another model.
            value = os.getenv('KADAN_H3_DEVICES', '0,1')
            if value not in ('0', '1', '0,1', '1,0'):
                raise InferenceFailure('Invalid KADAN_H3_DEVICES.')
            devices = [int(device) for device in value.split(',')]
            if self._devices is not None and self._devices != value:
                self._close_locked()
            resources = self.resources or chat_runtime.ensure_resources()
            resources.offload_inactive_devices('video', cancel)
            try:
                with ExitStack() as leases:
                    if self._context is None:
                        self._context = resources.reserve(self._owner + ':context', 'video',
                            host_bytes=256*1024**2, device_bytes={i: CONTEXT_BUDGET for i in devices},
                            evict=self._evict, cancel_event=cancel)
                    leases.enter_context(self._context.lease(cancel))
                    if spec is not None:
                        self._execution = resources.reserve(self._owner + ':execution', 'video',
                            host_bytes=HOST_BUDGET, device_bytes={i: DEVICE_BUDGET for i in devices},
                            cancel_event=cancel)
                        leases.enter_context(self._execution.lease(cancel))
                    if self._worker is None:
                        self._worker = self.process_factory()
                        self._worker.start(command, env=dict(os.environ, KADAN_H3_DEVICES=value))
                        self._devices = value
                    if spec is None:
                        return
                    if output.exists() or output.is_symlink():
                        raise FileExistsError(output)
                    self._workspace = Path(tempfile.mkdtemp(prefix='.h3-', dir=output.parent))
                    raw, encoded = self._workspace / 'video.y4m', self._workspace / 'video.mp4'
                    response = self._worker.exchange(dict(prompt=spec.prompt, output=str(raw),
                        short_edge=int(spec.resolution[:-1]), aspect=spec.aspect,
                        duration=spec.duration, updates=4, seed=spec.seed), 24*3600, cancel)
                    frames = validate_artifact(response, raw, spec)
                    self._encode(raw, encoded, frames, cancel)
                    check_cancel(cancel)
                    os.link(encoded, output)  # Same filesystem, exclusive publication.
                self._execution.release()
                self._execution = None
                shutil.rmtree(self._workspace)
                self._workspace = None
            except BaseException as error:
                self._close_locked()  # Leases have exited; release only after confirmed reap.
                if isinstance(error, InterruptedError):
                    raise ResourceCancelled('Video generation cancelled') from error
                raise
        finally:
            self._lock.release()
