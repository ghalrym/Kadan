"""Bounded complete conversation state, owned by the LLM wrapper.

Retain an admitted device state when headroom permits; otherwise copy to CPU
only after generation. Restores are isolated working copies under the request
lease. No hybrid state is cropped or rewound.
"""
from collections import OrderedDict
import copy
import hashlib
import json
from functools import wraps
from dataclasses import dataclass
import threading
from uuid import uuid4

import torch

from api.inference.resources import ResourceBusy, ResourceExhausted


def tensors_in(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors_in(item, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors_in(item, seen)
    elif type(value).__module__.startswith('transformers.') and hasattr(value, '__dict__'):
        for item in vars(value).values():
            yield from tensors_in(item, seen)


def clone_state(value, device, memo=None):
    """Copy cache tensors directly to destination, including hybrid conv/recurrent state.

    Avoid deepcopy on CUDA: it would first allocate an unbudgeted second GPU cache.
    Only Transformers cache objects and ordinary immutable/container values are accepted.
    """
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        result = value.detach().to(device=device, copy=True)
    elif isinstance(value, torch.device):
        return torch.device(device)
    elif isinstance(value, dict):
        result = {}
        memo[id(value)] = result
        result.update({key: clone_state(item, device, memo) for key, item in value.items()})
    elif isinstance(value, list):
        result = []
        memo[id(value)] = result
        result.extend(clone_state(item, device, memo) for item in value)
    elif isinstance(value, tuple):
        result = tuple(clone_state(item, device, memo) for item in value)
    elif type(value).__module__.startswith('transformers.') and hasattr(value, '__dict__'):
        result = copy.copy(value)
        memo[id(value)] = result
        for name, item in vars(value).items():
            setattr(result, name, clone_state(item, device, memo))
    elif value is None or isinstance(value, (str, int, float, bool, torch.dtype, type)) or callable(value):
        return value
    else:
        raise TypeError(f'Unsupported cache metadata: {type(value).__name__}')
    memo[id(value)] = result
    return result


def locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return call


@dataclass
class PrefixEntry:
    conversation: str
    identity: str
    tokens: tuple
    size: int
    state: object = None
    reservation: object = None
    live: bool = True
    host_bytes: int = 0
    device_bytes: dict | None = None
    history_count: int = 0
    history_digest: str | None = None


def history_digest(history):
    return hashlib.sha256(json.dumps(history, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


class ConversationCache:
    def __init__(self, resources, *, max_bytes=2 * 1024**3, max_entries=4):
        if max_bytes < 1 or max_entries < 1:
            raise ValueError('Conversation cache limits must be positive')
        self.resources, self.max_bytes, self.max_entries = resources, max_bytes, max_entries
        self.entries = OrderedDict()
        self.pending = {}
        self.owner = 'llm:conversation:' + uuid4().hex
        self.lock = threading.RLock()

    def _drop(self, entry):
        # Eviction must not invert admission/cache lock ordering.
        if not self.lock.acquire(blocking=False):
            raise ResourceBusy('Conversation cache is changing ownership')
        try:
            entry.state = None
            entry.live = False
            if entry.reservation is not None:
                entry.reservation.release()
                entry.reservation = None
            for store in (self.entries, self.pending):
                if store.get(entry.conversation) is entry:
                    del store[entry.conversation]
        finally:
            self.lock.release()

    @locked
    def invalidate(self, conversation):
        for store in (self.entries, self.pending):
            entry = store.get(conversation)
            if entry is not None:
                self._drop(entry)

    @locked
    def clear(self):
        for entry in list(self.entries.values()) + list(self.pending.values()):
            self._drop(entry)

    @locked
    def restore(self, conversation, tokens, identity, device, max_prefix, history=None):
        entry = self.entries.get(conversation)
        if entry is None:
            return None, 0, 'miss'
        history_changed = (history is not None and entry.history_digest is not None and
            (len(history) < entry.history_count or history_digest(history[:entry.history_count]) != entry.history_digest))
        if history_changed or entry.identity != identity or len(entry.tokens) > max_prefix or tuple(tokens[:len(entry.tokens)]) != entry.tokens:
            self.invalidate(conversation)
            return None, 0, 'invalidated'
        try:
            with entry.reservation.lease():
                state = clone_state(entry.state, device)
            self.entries.move_to_end(conversation)
            return state, len(entry.tokens), 'hit'
        except ResourceBusy:
            return None, 0, 'evicted'

    @locked
    def capture(self, conversation, tokens, identity, state, *, adopt=False):
        tensors = list(tensors_in(state))
        tensor_bytes = sum(t.numel() * t.element_size() for t in tensors)
        metadata_bytes = len(tokens) * 40 + 65536
        size = tensor_bytes + metadata_bytes
        self.invalidate(conversation)
        if size > self.max_bytes:
            return 'limit_exceeded'
        retained = lambda: list(self.entries.values()) + list(self.pending.values())
        while retained() and (len(retained()) >= self.max_entries or
                sum(entry.size for entry in retained()) + size > self.max_bytes):
            self._drop(retained()[0])
        devices = {t.device for t in tensors}
        device = next(iter(devices)) if len(devices) == 1 else torch.device('cpu')
        # Prefer already allocated device state only when conservative admission
        # fits alongside the still-active request. Admission rechecks the sampled
        # headroom. CPU fallback happens after the last token.
        keep_device = (adopt and device.type == 'cuda' and
            self.resources.available_devices().get(device.index or 0, 0) >= tensor_bytes)
        destinations = [device, torch.device('cpu')] if keep_device else [torch.device('cpu')]
        for destination in destinations:
            entry = PrefixEntry(conversation, identity, tuple(tokens), size)
            entry.host_bytes = metadata_bytes if destination.type == 'cuda' else size
            entry.device_bytes = {destination.index or 0: tensor_bytes} if destination.type == 'cuda' else {}
            try:
                entry.reservation = self.resources.reserve(self.owner + ':' + uuid4().hex, 'llm',
                    host_bytes=entry.host_bytes, device_bytes=entry.device_bytes, evict=lambda e=entry: self._drop(e))
                with entry.reservation.lease():
                    entry.state = state if adopt and devices == {destination} else clone_state(state, destination)
                    self.pending[conversation] = entry
                return 'retained'
            except (ResourceBusy, ResourceExhausted):
                self._drop(entry)
            except BaseException:
                self._drop(entry)
                raise
        return 'memory_pressure'

    @locked
    def commit(self, conversation, history=None, retention_reason=None):
        entry = self.pending.pop(conversation, None)
        if entry is not None and entry.live:
            self.entries[conversation] = entry
        entry = self.entries.get(conversation)
        if entry is not None and history is not None:
            entry.history_count, entry.history_digest = len(history), history_digest(history)
        return {'stored_tokens': len(entry.tokens) if entry else 0,
                'host_bytes': entry.host_bytes if entry else 0,
                'device_bytes': entry.device_bytes if entry else {},
                'retention_reason': ('evicted' if retention_reason == 'retained' and entry is None
                                     else retention_reason or ('retained' if entry else 'miss')),
                'limit_bytes': self.max_bytes}
