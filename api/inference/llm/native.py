"""Opt-in original C++/CUDA worker adapter; no native execution on module import.

The Python parent owns global admission and tokenization. The child owns one
bounded arena and does greedy token steps; its accounting is a sub-budget.
"""
from collections.abc import Mapping
from contextlib import contextmanager
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import threading
import time
from uuid import uuid4

from api.inference.llm.context import ContextLimitError, ContextMemoryError, resolve_context
from api.inference.placement import select_device
from api.inference.resources import ResourceBusy, ResourceExhausted

MIB = 1024**2
METADATA_BYTES = 256 * MIB
NATIVE_HOST_BYTES = 258 * MIB
PYTHON_HOST_BYTES = 512 * MIB
HEADROOM_BYTES = 512 * MIB
MAX_FRAME = 4096


class NativeProtocolError(RuntimeError):
    pass


def check_cancel(event):
    if event is not None and event.is_set():
        raise InterruptedError('Native inference cancelled')


def numbers(line, prefix, count):
    parts = line.split()
    if parts[:len(prefix)] != prefix or len(parts) != len(prefix) + count:
        raise NativeProtocolError('Invalid native worker response')
    values = parts[len(prefix):]
    if any(not item.isascii() or not item.isdecimal() or len(item) > 20 for item in values):
        raise NativeProtocolError('Invalid native worker integer')
    return [int(item) for item in values]


class WorkerProcess:
    """Single-owner bounded IPC, draining stderr while awaiting every reply."""
    def __init__(self):
        # Construct without side effects. The adapter must own this handle
        # before start() can spawn or perform fallible pipe/selector setup.
        self.process = self.selector = None
        self.buffer = bytearray()
        self.diagnostics = bytearray()
        self.closed = False
        self.io_ready = False

    def start(self, command):
        if self.process is not None or self.closed:
            raise RuntimeError('Native process handle cannot be reused')
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, bufsize=0, start_new_session=True)
        self.selector = selectors.DefaultSelector()
        for stream, kind in ((self.process.stdout, 'out'), (self.process.stderr, 'err')):
            os.set_blocking(stream.fileno(), False)
            self.selector.register(stream, selectors.EVENT_READ, kind)
        self.io_ready = True

    def _pump(self, deadline, cancel):
        check_cancel(cancel)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Native worker deadline expired')
        for key, _ in self.selector.select(min(.05, remaining)):
            data = os.read(key.fileobj.fileno(), 4096)
            if not data:
                self.selector.unregister(key.fileobj)
            elif key.data == 'err':
                self.diagnostics.extend(data)
                del self.diagnostics[:-8192]
            else:
                self.buffer.extend(data)
                if len(self.buffer) > MAX_FRAME:
                    raise NativeProtocolError('Native worker frame too large')

    def read(self, timeout, cancel=None):
        deadline = time.monotonic() + timeout
        while b'\n' not in self.buffer:
            if not self.selector.get_map():
                raise NativeProtocolError('Native worker exited without a complete reply')
            self._pump(deadline, cancel)
        line, _, rest = self.buffer.partition(b'\n')
        self.buffer = bytearray(rest)
        try:
            text = line.decode('ascii')
        except UnicodeDecodeError as error:
            raise NativeProtocolError('Non-ASCII native worker frame') from error
        if text.startswith('error '):
            raise NativeProtocolError('Native worker failed: ' + self.diagnostics.decode('utf-8', errors='replace')[-1000:])
        return text

    def exchange(self, command, timeout, cancel=None):
        check_cancel(cancel)
        if self.buffer or self.process.poll() is not None:
            raise NativeProtocolError('Native worker unavailable or sent unsolicited data')
        data = (command + '\n').encode('ascii')
        if len(data) > 128:
            raise NativeProtocolError('Native command too large')
        # One small command at a time; no pipelining can fill this pipe.
        self.process.stdin.write(data)
        return self.read(timeout, cancel)

    def finish(self, timeout=5):
        deadline = time.monotonic() + timeout
        while self.process.poll() is None or self.selector.get_map():
            self._pump(deadline, None)
            if self.buffer:
                raise NativeProtocolError('Unexpected trailing worker output')
        if self.process.returncode != 0:
            raise NativeProtocolError('Native worker exit was not successful')

    def stop(self):
        """Return only after owned child exit; on wait failure retain parent accounting.

        The reviewed native worker never forks. Signal its owned process group
        while the leader is live; never signal a reaped/reusable PID.
        """
        if self.closed:
            return
        if self.process is None:
            self.closed = True
            return
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=5)  # Failure propagates: no release claim.
        finally:
            if self.process.poll() is not None:
                if self.selector is not None:
                    self.selector.close()
                for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                    stream.close()
                self.closed = True


def load_tokenizer(path):
    # Optional dependency stays lazy so IPC/fault tests need no Torch/Transformers;
    # selecting the native backend never imports a Python numerical model engine.
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def text_streamer(tokenizer, emit):
    # Reuse the existing API text/UTF-8 buffering behavior only when streaming.
    import torch
    from api.inference.llm.streaming import TextEvents
    events = TextEvents(tokenizer, emit)
    class Stream:
        def put(self, token):
            events.put(torch.tensor([token], device='cpu'))
        def end(self):
            events.end()
    return Stream()


class NativeAdapter:
    def __init__(self, entry, path, resources, device='auto', cancel_event=None,
                 *, tokenizer_factory=load_tokenizer, streamer_factory=text_streamer,
                 worker_path=None, load_timeout=1800, step_timeout=300):
        if entry.id != 'small':
            raise ValueError('The native backend currently supports only the small Qwen checkpoint')
        self.root = Path(path).resolve()
        config_path = self.root / 'config.json'
        if config_path.stat().st_size > 65536:
            raise ValueError('Native checkpoint config is too large')
        self.config = json.loads(config_path.read_text())
        if self.config.get('model_type') != 'qwen3_5_moe':
            raise ValueError('Native backend requires qwen3_5_moe metadata')
        self.binary = Path(worker_path or os.environ.get('KADAN_NATIVE_WORKER', '/opt/kadan/bin/kadan-model-worker'))
        if not self.binary.is_absolute() or not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise ValueError('KADAN_NATIVE_WORKER must name an installed absolute executable path')
        self.resources, self.device, self.cancel = resources, device, cancel_event
        self.tokenizer_factory, self.streamer_factory = tokenizer_factory, streamer_factory
        self.load_timeout, self.step_timeout = load_timeout, step_timeout
        self.owner = 'native-llm:' + uuid4().hex
        self.host = self.reservation = self.worker = self.tokenizer = None
        self.capacity = self.vocabulary = self.arena = 0
        self._lock = threading.Lock()
        self._closed = False

    @property
    def is_resident(self):
        return (self.worker is not None and self.worker.io_ready and not self.worker.closed
                and self.worker.process.poll() is None)

    def configure_context(self, configured):
        with self._lock:
            self._configure_context_locked(configured)

    def _configure_context_locked(self, configured):
        if self._closed:
            raise RuntimeError('Native adapter is closed')
        supported, effective = resolve_context(self.config, configured)
        if effective > 262144:
            raise ContextLimitError('Native worker context exceeds its reviewed 262144-token bound')
        if self.worker is not None:
            if effective != self.capacity:
                raise ContextLimitError('Unload native worker before changing its context')
            return
        self.configured_context_limit = configured
        self.supported_context_limit = supported
        self.effective_context_limit = effective
        self.capacity = effective
        try:
            self.host = self.resources.reserve(self.owner + ':host', 'llm',
                host_bytes=NATIVE_HOST_BYTES + PYTHON_HOST_BYTES, evict=self._evict, cancel_event=self.cancel)
            with self.host.lease(self.cancel):
                planner = WorkerProcess()
                # Publish ownership before spawn and fallible IPC setup.
                self.worker = planner
                planner.start([str(self.binary), '--plan', str(self.root), str(effective), str(METADATA_BYTES)])
                host, arena, vocab, capacity, staging = numbers(planner.read(self.load_timeout, self.cancel), ['plan', '1'], 5)
                planner.finish()
                planner.stop()
                self.worker = None
                if host != NATIVE_HOST_BYTES or capacity != effective or not 0 < vocab <= 262144 or not 0 < arena or not 0 < staging <= MIB:
                    raise NativeProtocolError('Native plan violates the adapter bounds')
                self.vocabulary, self.arena = vocab, arena
                self.device = select_device(self.resources, arena + HEADROOM_BYTES, self.device)
                index = int(self.device[5:])
                self.reservation = self.resources.reserve(self.owner + ':device', 'llm',
                    device_bytes={index: arena + HEADROOM_BYTES}, evict=self._evict, cancel_event=self.cancel)
                with self.reservation.lease(self.cancel):
                    self.tokenizer = self.tokenizer_factory(self.root)
                    self.worker = WorkerProcess()
                    self.worker.start([str(self.binary), '--serve', str(self.root), str(index),
                        str(effective), str(host), str(arena + HEADROOM_BYTES), str(HEADROOM_BYTES)])
                    ready = numbers(self.worker.read(self.load_timeout, self.cancel), ['ready', '1'], 4)
                    if ready != [vocab, effective, arena, host]:
                        raise NativeProtocolError('Native worker readiness differs from admitted plan')
        except BaseException:
            self._close_locked()
            raise

    def _close_locked(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        self.tokenizer = None
        for name in ('reservation', 'host'):
            handle = getattr(self, name)
            if handle is not None:
                handle.release()
                setattr(self, name, None)

    def _evict(self):
        if not self._lock.acquire(blocking=False):
            raise ResourceBusy('Native worker is active')
        try:
            self._close_locked()
        finally:
            self._lock.release()

    def close(self):
        with self._lock:
            self._closed = True
            # Normal lifecycle requests a clean child close; failure still stops
            # and reaps it before releasing global ownership.
            try:
                if self.worker_alive:
                    if self.worker.exchange(self._close_command(), 5) != self._close_reply():
                        raise NativeProtocolError('Native close did not confirm zero reservations')
                    self.worker.finish()
            finally:
                self._close_locked()

    @property
    def worker_alive(self):
        return (self.worker is not None and self.worker.io_ready and not self.worker.closed
                and self.worker.process.poll() is None)

    def _close_command(self):
        return 'close'

    def _close_reply(self):
        return 'closed 0'

    def _restore_locked(self, cancel):
        pass

    @contextmanager
    def _leases(self, cancel):
        with self.host.lease(cancel), self.reservation.lease(cancel):
            yield

    def _begin_request(self, cancel):
        if self.worker.exchange('reset', self.step_timeout, cancel) != 'ok reset':
            raise NativeProtocolError('Native reset failed')

    def _end_request(self, cancel):
        pass

    def _step(self, token, stop, expected, cancel):
        selected, eos, progress = numbers(self.worker.exchange(f'step {token} {int(stop)}', self.step_timeout, cancel), ['token'], 3)
        if selected >= self.vocabulary or eos not in (0,1) or progress != expected:
            raise NativeProtocolError('Native step violates vocabulary/EOS/progress contract')
        return selected, bool(eos)

    def generate(self, messages, max_new_tokens=256, cancel_event=None, on_event=None, conversation_id=None):
        # Eviction, reload and generation share one ownership gate. There is
        # no unlocked window in which a newly restored worker can be evicted.
        with self._lock:
            if self.worker is None and not self._closed:
                self.cancel = cancel_event
                try:
                    self._configure_context_locked(self.configured_context_limit)
                except (ResourceExhausted, ResourceBusy) as error:
                    # Retry only after confirmed cleanup, never uncertain exit.
                    if self.worker is None and self.host is None and self.reservation is None:
                        raise ContextMemoryError(str(error)) from error
                    raise
            check_cancel(cancel_event)
            self._restore_locked(cancel_event)
            if not self.is_resident:
                raise RuntimeError('Native worker is unloaded; explicitly load it again')
            if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 1024:
                raise ContextLimitError('Output token limit must be between 1 and 1024')
            chat = [{'role': m['role'], 'content': m.get('text', m.get('content', ''))} for m in messages]
            with self._leases(cancel_event):
                tokens = self.tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True,
                    enable_thinking=False, preserve_thinking=True)
                if isinstance(tokens, Mapping):
                    tokens = tokens['input_ids']
                if not isinstance(tokens, list) or not tokens or any(type(t) is not int or not 0 <= t < self.vocabulary for t in tokens):
                    raise ContextLimitError('Tokenizer returned invalid native input IDs')
                if len(tokens) + max_new_tokens > self.capacity:
                    raise ContextLimitError(f'Prompt ({len(tokens)}) plus output budget ({max_new_tokens}) exceeds native context {self.capacity}; nothing was truncated')
                generated = []
                streamer = self.streamer_factory(self.tokenizer, on_event) if on_event else None
                started = time.monotonic()
                first = last = None
                reason = 'length'
                try:
                    self._begin_request(cancel_event)
                    for index, token in enumerate(tokens):
                        selected, eos = self._step(token, False, index+1, cancel_event)
                    for index in range(max_new_tokens):
                        now = time.monotonic()
                        if eos:
                            reason = 'stop'
                            break
                        generated.append(selected)
                        first = now if first is None else first
                        last = now
                        if streamer is not None:
                            streamer.put(selected)
                        if index + 1 < max_new_tokens:
                            selected, eos = self._step(selected, True, len(tokens)+index+1, cancel_event)
                    check_cancel(cancel_event)
                    self._end_request(cancel_event)
                    result = self.tokenizer.decode(generated, skip_special_tokens=True)
                    if streamer is not None:
                        streamer.end()
                    if on_event:
                        seconds = last-first if len(generated)>1 else 0
                        on_event({'timing': {'generation_ttft_ms': (first-started)*1000 if first else None,
                            'prefill_ms': (first-started)*1000 if first else None,
                            'decode_tokens_per_second': (len(generated)-1)/seconds if seconds>0 else None,
                            'output_tokens': len(generated), 'prefill_tokens': len(tokens)}})
                        if conversation_id:
                            on_event({'cache': {'hit': False, 'reused_tokens': 0, 'stored_tokens': 0,
                                'host_bytes': 0, 'device_bytes': {}, 'reason': 'native_prefix_reuse_unavailable',
                                'retention_reason': 'disabled', 'limit_bytes': 0}})
                        on_event({'finish_reason': reason})
                    return result
                except BaseException:
                    # Do not reuse an uncertain sequence or a timed-out worker.
                    self.worker.stop()
                    raise


def build_native(entry, path, resources, device='auto', cancel_event=None):
    return NativeAdapter(entry, path, resources, device, cancel_event)
