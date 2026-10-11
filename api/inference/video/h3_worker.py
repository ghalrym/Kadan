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
import wave

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
PROCESS_HOST_BUDGET = HOST_BUDGET + 256 * 1024**2
MAX_WEIGHT_CACHE_BYTES = 32 * GIB
DEVICE_WEIGHT_METADATA_BYTES = 4 * 1024**2



def h3_host_budget(spec):
    """Admit bounded activations, streamed weights and packing, not all weights.

    Use the tokenizer limit before tokenization, so this plan reads no tensors.
    Turbo's largest projection adds 116224 bytes/token to the denoiser's
    379904 bytes/token. Decoder phases run sequentially and reuse the envelope.
    """
    if spec is None:
        return HOST_BUDGET
    sizes = {'480p': {'16:9': (864, 480), '9:16': (480, 864), '1:1': (480, 480)},
             '768p': {'16:9': (1344, 768), '9:16': (768, 1344), '1:1': (768, 768)}}
    width, height = sizes[spec.resolution][spec.aspect]
    frames = ((spec.duration * 24 - 5 + 16) // 17) * 17 + 5
    t, h, w = (frames - 5) // 17 * 5 + 2, height // 16, width // 16
    nv, na = t * h * w // 4, 2 * ((frames * 5 + 1) // 3)
    # Byte-level BPE emits no more tokens than normalized UTF-8 bytes.
    # NFC canonical decomposition expands UTF-8 by less than fourfold;
    # added tokens consume at least one byte and insert no extra tokens.
    # Match the native 128000-token bound without imposing a 512-token limit.
    nt = min(128000, 4 * len(spec.prompt.encode('utf-8')))
    caller = (32 * 1024**2 + (3 * nv * 96 + 2 * na * 32 + 7 * h * w * 24) * 4
              + width * height * 399 + (nt + nv + na) * 20 + nt * 5120 * 4)
    patches, tokens = 7 * h * w, 7 * h * w + 5
    vae = 4 * (tokens * (4099 + 14336 + 16384) + patches * 3072)
    execution = max(496128 * (nt + nv + na), vae, 512 * 1024**2)
    # Includes parsers, single-head attention packing, CUDA source tiles,
    # weight-bank metadata and allocator slack; optional host cache is separate.
    return ((execution + caller + 4 * GIB + GIB - 1) // GIB) * GIB


def h3_gpu_budgets(resources, devices):
    capacity = resources.snapshot()['device_capacity_bytes']
    raw = os.environ.get('KADAN_H3_GPU_BUDGET_BYTES')
    if raw is None:
        available = resources.available_devices(reclaim=True)
        budgets = {d: min(24 * GIB, capacity.get(d, 0),
                          max(CONTEXT_BUDGET + DEVICE_BUDGET, available.get(d, 0) - 64 * 1024**2))
                   for d in devices}
    elif raw.isascii() and raw.isdecimal():
        budgets = {d: int(raw) for d in devices}
    else:
        raise InferenceFailure('Invalid H3 GPU budget.')
    if any(not CONTEXT_BUDGET + DEVICE_BUDGET <= b <= 24 * GIB or capacity.get(d, 0) < b for d, b in budgets.items()):
        raise InferenceFailure('H3 GPU budget exceeds usable capacity or cannot hold execution scratch.')
    return budgets


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
                    self._capture_diagnostics(data)
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


def resolve_command(model_id, cancel=None):
    binary = Path(os.getenv('KADAN_NATIVE_H3_WORKER', '/opt/kadan/bin/kadan-h3-worker')).expanduser()
    if not binary.is_absolute() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise InferenceFailure('Native H3 worker is unavailable; configure KADAN_NATIVE_H3_WORKER.')
    # This mode only reports compile-time capabilities: no model or CUDA context.
    result = subprocess.run([str(binary), '--capabilities'], capture_output=True, timeout=10, check=True)
    expected = dict(protocol=2, cuda=True, audio=True, host_budget=HOST_BUDGET, device_budget=DEVICE_BUDGET)
    if decode_response(result.stdout) != expected:
        raise InferenceFailure('H3 requires the CUDA native worker with protocol 2.')
    entry, checkpoint = (model_manager.ensure_checkpoint(model_id, cancel) if cancel is not None
                         else model_manager.get_checkpoint(model_id))
    if entry.revision != H3_INT8_REVISION:
        raise InferenceFailure('H3 checkpoint revision does not match the native worker.')
    names = ('FL2VA/tokenizer/tokenizer.json', 'FL2VA/text_encoder/model.safetensors',
             'FL2VA/transformer/model.safetensors',
             'loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors',
             'FL2VA/video_vae/model.safetensors', 'FL2VA/audio_vae/model.safetensors')
    paths = [checkpoint / name for name in names]
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        raise InferenceFailure('H3 checkpoint is incomplete.')
    if shutil.which('ffmpeg') is None:
        raise InferenceFailure('H3 requires the FFmpeg codec executable.')
    return [str(binary), *(str(path) for path in paths)]


def validate_artifact(response, raw, spec, cache_budget=0, device_budgets=None):
    fields = {'output', 'width', 'height', 'frames', 'audio', 'resident_bytes', 'device_resident_bytes', 'weight_cache_bytes', 'device_weight_metadata_bytes'}
    if not isinstance(response, dict) or set(response) != fields:
        raise LineProtocolError('H3 generation failed: ' + str(response)[:200])
    device_budgets = device_budgets or {}
    retained = response['device_resident_bytes']
    if (not isinstance(retained, list) or len(retained) != 2 or any(type(n) is not int
            or not 0 <= n <= device_budgets.get(i, 0) for i, n in enumerate(retained))):
        raise LineProtocolError('H3 device memory accounting mismatch')
    metadata = DEVICE_WEIGHT_METADATA_BYTES * sum(n > 0 for n in retained)
    if (response['output'] != str(raw) or response['audio'] is not True
            or type(response['weight_cache_bytes']) is not int or not 0 <= response['weight_cache_bytes'] <= cache_budget
            or type(response['device_weight_metadata_bytes']) is not int or response['device_weight_metadata_bytes'] != metadata
            or type(response['resident_bytes']) is not int or response['resident_bytes'] != response['weight_cache_bytes'] + metadata):
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
    audio = Path(str(raw) + '.wav')
    expected_samples = round(frames * 5 / 3) * 800
    if audio.is_symlink() or not audio.is_file() or audio.stat().st_size != 44 + expected_samples * 4:
        raise LineProtocolError('H3 audio artifact is missing or truncated')
    try:
        with wave.open(str(audio), 'rb') as source:
            if (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getnframes()) != (2, 2, 32000, expected_samples):
                raise LineProtocolError('H3 audio format or duration mismatch')
    except (wave.Error, EOFError) as error:
        raise LineProtocolError('H3 audio header is invalid') from error
    return frames


class H3Provider:
    def __init__(self, model_id='h3-fl2va-int8-turbo', *, resources=None,
                 resolve=resolve_command, process_factory=H3Process):
        self.model_id, self.resources = model_id, resources
        self.resolve, self.process_factory = resolve, process_factory
        self._lock = threading.Lock()
        self._worker = self._codec = self._host = self._context = self._execution = self._workspace = None
        self._quarantined = False
        self._devices = None
        self._gpu_budget = None
        self._cache_budget = 0
        self._host_budget = None
        self._parked = False
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
        for field in ('_execution', '_context', '_host'):
            reservation = getattr(self, field)
            if reservation is not None:
                reservation.release()
                setattr(self, field, None)
        if self._workspace is not None:
            shutil.rmtree(self._workspace)
            self._workspace = None
        self._quarantined = False
        self._devices = None
        self._gpu_budget = None
        self._host_budget = None
        self._parked = False

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
        (self.resources or chat_runtime.ensure_resources()).offload_workload_devices('video', cancellation)

    def _park(self):
        if not self._lock.acquire(blocking=False):
            raise ResourceBusy('H3 is executing')
        try:
            if self._worker is not None and not self._parked:
                try:
                    self._control('park', 'parked', threading.Event())
                except BaseException:
                    self._close_locked()  # Failed ACK requires confirmed reap.
                    raise
                self._parked = True
            if self._context is not None:
                self._context.release()
                self._context = None
        finally:
            self._lock.release()

    def _control(self, command, state, cancel):
        response = self._worker.exchange({'control': command}, 60, cancel)
        if (not isinstance(response, dict) or set(response) != {'state', 'resident_bytes', 'device_resident_bytes'}
                or response['state'] != state or type(response['resident_bytes']) is not int
                or not 0 <= response['resident_bytes'] <= self._cache_budget
                or response['device_resident_bytes'] != [0, 0]
                or any(type(n) is not int for n in response['device_resident_bytes'])):
            raise LineProtocolError('H3 device handoff was not acknowledged')

    def load(self, cancellation):
        self._use(None, None, cancellation)

    def generate(self, spec, output_path, cancellation):
        self.validate(spec)
        self._use(spec, Path(output_path), cancellation)

    def _encode(self, raw, target, frames, cancel):
        check_cancel(cancel)
        self._codec = LineProtocolProcess()
        self._codec.start([shutil.which('ffmpeg'), '-nostdin', '-v', 'error', '-n',
            '-threads', '1', '-i', str(raw), '-i', str(raw) + '.wav',
            '-map', '0:v:0', '-map', '1:a:0', '-c:a', 'aac', '-b:a', '192k',
            '-c:v', 'libx264', '-threads', '1',
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
            command = self.resolve(self.model_id, cancel)
            resources = self.resources or chat_runtime.ensure_resources()
            value = os.getenv('KADAN_H3_DEVICES', 'auto')
            if value == 'auto':
                capacity = resources.snapshot()['device_capacity_bytes']
                devices = [d for d in (0, 1) if capacity.get(d, 0) >= CONTEXT_BUDGET + DEVICE_BUDGET]
                if not devices:
                    raise InferenceFailure('No supported CUDA device can hold H3 execution scratch and context.')
                value = ','.join(map(str, devices))
            elif value in ('0', '1', '0,1', '1,0'):
                devices = [int(device) for device in value.split(',')]
            else:
                raise InferenceFailure('Invalid KADAN_H3_DEVICES.')
            gpu_budget = h3_gpu_budgets(resources, devices)
            host_budget = h3_host_budget(spec)
            if self._host_budget is not None and self._host_budget != host_budget:
                self._close_locked()
            if self._devices is not None and (self._devices != value or self._gpu_budget != gpu_budget):
                self._close_locked()
            resources.offload_inactive_devices('video', cancel)
            try:
                with ExitStack() as leases:
                    # A zero tensor ledger does not prove allocator arenas or CUDA
                    # host caches returned their pages. Retain the full process host
                    # envelope until reaping, including while its GPU work is idle.
                    if self._host is None:
                        # Optional cache uses fresh headroom and leaves allocator/OS slack.
                        # A required execution envelope can wait; the cache may be zero.
                        self._cache_budget = min(MAX_WEIGHT_CACHE_BYTES, max(0,
                            resources.available_host() - host_budget - 512 * 1024**2))
                        self._host = resources.reserve(self._owner + ':host', 'video',
                            host_bytes=host_budget + 256 * 1024**2 + self._cache_budget,
                            evict=self._evict, cancel_event=cancel)
                    leases.enter_context(self._host.lease(cancel))
                    if self._context is None:
                        self._context = resources.reserve(self._owner + ':context', 'video',
                            device_bytes={i: gpu_budget[i] - DEVICE_BUDGET for i in devices},
                            evict=self._park, cancel_event=cancel)
                    leases.enter_context(self._context.lease(cancel))
                    if spec is not None:
                        self._execution = resources.reserve(self._owner + ':execution', 'video',
                            device_bytes={i: DEVICE_BUDGET for i in devices},
                            cancel_event=cancel)
                        leases.enter_context(self._execution.lease(cancel))
                    if self._worker is None:
                        self._worker = self.process_factory()
                        self._worker.start(command, env=dict(os.environ, KADAN_H3_DEVICES=value,
                            KADAN_H3_HOST_BUDGET_BYTES=str(host_budget),
                            KADAN_H3_WEIGHT_CACHE_BYTES=str(self._cache_budget), KADAN_H3_GPU_BUDGET_BYTES=str(min(gpu_budget.values())),
                            KADAN_H3_GPU_BUDGETS=','.join(f'{d}:{b}' for d, b in gpu_budget.items())))
                        self._host_budget = host_budget
                        self._devices = value
                        self._gpu_budget = gpu_budget
                    if self._parked:
                        self._control('resume', 'resumed', cancel)
                        self._parked = False
                    if spec is None:
                        return
                    if output.exists() or output.is_symlink():
                        raise FileExistsError(output)
                    self._workspace = Path(tempfile.mkdtemp(prefix='.h3-', dir=output.parent))
                    raw, encoded = self._workspace / 'video.y4m', self._workspace / 'video.mp4'
                    response = self._worker.exchange(dict(prompt=spec.prompt, output=str(raw),
                        short_edge=int(spec.resolution[:-1]), aspect=spec.aspect,
                        duration=spec.duration, updates=4, seed=spec.seed), 24*3600, cancel)
                    frames = validate_artifact(response, raw, spec, self._cache_budget,
                        {i: gpu_budget[i] - CONTEXT_BUDGET - DEVICE_BUDGET for i in devices})
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
