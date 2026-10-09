"""Opt-in original C++/CUDA worker adapter; no model execution on module import.

The Python parent owns global admission and tokenization. The child owns one
bounded arena and does greedy token steps; its accounting is a sub-budget.
"""
from collections.abc import Mapping
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from uuid import uuid4

from api.inference.llm.context import ContextLimitError, ContextMemoryError, resolve_context
from api.inference.line_protocol import LineProtocolError, LineProtocolProcess, check_cancel
from api.inference.placement import select_device
from api.inference.resources import ResourceBusy, ResourceExhausted

MIB = 1024**2
METADATA_BYTES = 256 * MIB
QWEN_HOST_BYTES = 258 * MIB
PYTHON_HOST_BYTES = 512 * MIB
HEADROOM_BYTES = 512 * MIB


def numbers(line, prefix, count):
    parts = line.split()
    if parts[:len(prefix)] != prefix or len(parts) != len(prefix) + count:
        raise LineProtocolError('Invalid inference subprocess response')
    values = parts[len(prefix):]
    if any(not item.isascii() or not item.isdecimal() or len(item) > 20 for item in values):
        raise LineProtocolError('Invalid inference subprocess integer')
    return [int(item) for item in values]



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


class QwenSubprocessAdapter:
    def __init__(self, entry, path, resources, device='auto', cancel_event=None,
                 *, tokenizer_factory=load_tokenizer, streamer_factory=text_streamer,
                 worker_path=None, load_timeout=1800, step_timeout=300):
        if entry.id != 'small':
            raise ValueError('The Qwen subprocess currently supports only the small Qwen checkpoint')
        self.root = Path(path).resolve()
        config_path = self.root / 'config.json'
        if config_path.stat().st_size > 65536:
            raise ValueError('Qwen checkpoint config is too large')
        self.config = json.loads(config_path.read_text())
        if self.config.get('model_type') != 'qwen3_5_moe':
            raise ValueError('Qwen subprocess requires qwen3_5_moe metadata')
        self.binary = Path(worker_path or os.environ.get('KADAN_NATIVE_WORKER', '/opt/kadan/bin/kadan-model-worker'))
        if not self.binary.is_absolute() or not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise ValueError('KADAN_NATIVE_WORKER must name an installed absolute executable path')
        self.resources, self.device, self.cancel = resources, device, cancel_event
        self.tokenizer_factory, self.streamer_factory = tokenizer_factory, streamer_factory
        self.load_timeout, self.step_timeout = load_timeout, step_timeout
        self.owner = 'qwen-subprocess:' + uuid4().hex
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
            raise RuntimeError('Qwen adapter is closed')
        supported, effective = resolve_context(self.config, configured)
        if effective > 262144:
            raise ContextLimitError('Inference subprocess context exceeds its reviewed 262144-token bound')
        if self.worker is not None:
            if effective != self.capacity:
                raise ContextLimitError('Unload inference subprocess before changing its context')
            return
        self.configured_context_limit = configured
        self.supported_context_limit = supported
        self.effective_context_limit = effective
        self.capacity = effective
        try:
            self.host = self.resources.reserve(self.owner + ':host', 'llm',
                host_bytes=QWEN_HOST_BYTES + PYTHON_HOST_BYTES, evict=self._evict, cancel_event=self.cancel)
            with self.host.lease(self.cancel):
                planner = LineProtocolProcess()
                # Publish ownership before spawn and fallible IPC setup.
                self.worker = planner
                planner.start([str(self.binary), '--plan', str(self.root), str(effective), str(METADATA_BYTES)])
                host, arena, vocab, capacity, staging = numbers(planner.read(self.load_timeout, self.cancel), ['plan', '1'], 5)
                planner.finish()
                planner.stop()
                self.worker = None
                if host != QWEN_HOST_BYTES or capacity != effective or not 0 < vocab <= 262144 or not 0 < arena or not 0 < staging <= MIB:
                    raise LineProtocolError('Qwen plan violates the adapter bounds')
                self.vocabulary, self.arena = vocab, arena
                self.device = select_device(self.resources, arena + HEADROOM_BYTES, self.device)
                index = int(self.device[5:])
                self.reservation = self.resources.reserve(self.owner + ':device', 'llm',
                    device_bytes={index: arena + HEADROOM_BYTES}, evict=self._evict, cancel_event=self.cancel)
                with self.reservation.lease(self.cancel):
                    self.tokenizer = self.tokenizer_factory(self.root)
                    self.worker = LineProtocolProcess()
                    self.worker.start([str(self.binary), '--serve', str(self.root), str(index),
                        str(effective), str(host), str(arena + HEADROOM_BYTES), str(HEADROOM_BYTES)])
                    ready = numbers(self.worker.read(self.load_timeout, self.cancel), ['ready', '1'], 4)
                    if ready != [vocab, effective, arena, host]:
                        raise LineProtocolError('Inference subprocess readiness differs from admitted plan')
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
            raise ResourceBusy('Inference subprocess is active')
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
                if self.is_resident:
                    if self.worker.exchange('close', 5) != 'closed 0':
                        raise LineProtocolError('Qwen close did not confirm zero reservations')
                    self.worker.finish()
            finally:
                self._close_locked()

    def _step(self, token, stop, expected, cancel):
        selected, eos, progress = numbers(self.worker.exchange(f'step {token} {int(stop)}', self.step_timeout, cancel), ['token'], 3)
        if selected >= self.vocabulary or eos not in (0,1) or progress != expected:
            raise LineProtocolError('Qwen step violates vocabulary/EOS/progress contract')
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
            if not self.is_resident:
                raise RuntimeError('Inference subprocess is unloaded; explicitly load it again')
            if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 1024:
                raise ContextLimitError('Output token limit must be between 1 and 1024')
            chat = [{'role': m['role'], 'content': m.get('text', m.get('content', ''))} for m in messages]
            with self.host.lease(cancel_event), self.reservation.lease(cancel_event):
                tokens = self.tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True,
                    enable_thinking=False, preserve_thinking=True)
                if isinstance(tokens, Mapping):
                    tokens = tokens['input_ids']
                if not isinstance(tokens, list) or not tokens or any(type(t) is not int or not 0 <= t < self.vocabulary for t in tokens):
                    raise ContextLimitError('Tokenizer returned invalid Qwen input IDs')
                if len(tokens) + max_new_tokens > self.capacity:
                    raise ContextLimitError(f'Prompt ({len(tokens)}) plus output budget ({max_new_tokens}) exceeds Qwen context {self.capacity}; nothing was truncated')
                generated = []
                streamer = self.streamer_factory(self.tokenizer, on_event) if on_event else None
                started = time.monotonic()
                first = last = None
                reason = 'length'
                try:
                    if self.worker.exchange('reset', self.step_timeout, cancel_event) != 'ok reset':
                        raise LineProtocolError('Qwen reset failed')
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


def build_qwen_subprocess(entry, path, resources, device='auto', cancel_event=None):
    return QwenSubprocessAdapter(entry, path, resources, device, cancel_event)
