"""Owned host expert banks and a byte-bounded, event-safe device LRU.

Packed experts stay in host RAM. Only demanded experts enter the device cache.
Dense decode scratch, activations and model-resident tensors are separate budgets.
With a shared ResourceManager, each cached entry reserves its actual device bytes;
request KV growth may reclaim inactive entries without dropping the CPU bank.
The adapter separately reserves host staging (up to twice the largest expert).
The use() context serializes consumers and protects CUDA lifetimes on the current
stream. Callers must not retain device tensors beyond that context or launch work
on other streams. CUDA is optional: device='cpu' executes the same cache policy for
tiny deterministic tests, not as a claim of CUDA validation.
"""
from collections import OrderedDict
from collections.abc import Hashable, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from uuid import uuid4

from api.inference.resources import ResourceBusy

import torch
from torch import Tensor


def tensor_bytes(tensors: Mapping[str, Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in tensors.values())


class ExpertBank:
    """Retain immutable CPU tensor mappings; slice storage is shared, never cloned.

    Ownership is logical: callers must not mutate these tensors while bank/cache
    users exist. Pinning the whole model is deliberately avoided.
    """
    def __init__(self, experts: Mapping[Hashable, Mapping[str, Tensor]]):
        self._experts = {}
        storages = {}
        for key, tensors in experts.items():
            if not tensors or any(t.device.type != 'cpu' or t.requires_grad for t in tensors.values()):
                raise ValueError('ExpertBank requires nonempty CPU inference tensors.')
            self._experts[key] = MappingProxyType(dict(tensors))
            for tensor in tensors.values():
                storage = tensor.untyped_storage()
                storages[storage.data_ptr()] = storage.nbytes()
        self.host_bytes = sum(storages.values())
        self.max_expert_bytes = max((tensor_bytes(t) for t in self._experts.values()), default=0)

    def get(self, key: Hashable) -> Mapping[str, Tensor]:
        return self._experts[key]

    def __len__(self):
        return len(self._experts)


@dataclass
class _Entry:
    tensors: dict[str, Tensor]
    size: int
    finished: object = None
    reservation: object = None


class _Lease(Mapping):
    """Invalidate the mapping at context exit so a loop variable cannot pin evictees."""
    def __init__(self, tensors):
        self._tensors = tensors

    def _check(self):
        if self._tensors is None:
            raise RuntimeError('Expert lease has ended.')
        return self._tensors

    def __getitem__(self, key):
        return self._check()[key]

    def __iter__(self):
        return iter(self._check())

    def __len__(self):
        return len(self._check())


class ExpertCache:
    def __init__(self, bank: ExpertBank, capacity_bytes: int, device='cuda:0',
                 *, resources=None, owner=None, resource_device=None):
        if capacity_bytes <= 0:
            raise ValueError('Expert cache capacity must be positive.')
        self.bank = bank
        self.capacity_bytes = capacity_bytes
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Only CPU reference and CUDA expert caches are supported.')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; cannot create a GPU expert cache.')
        if resources is not None and not owner:
            raise ValueError('A shared-budget expert cache requires an owner.')
        self.resources = resources
        self._owner = f'{owner}:experts:{uuid4().hex}'
        self._next_entry = 0
        self._resource_device = resource_device if resource_device is not None else self.device.index
        if resources is not None and self._resource_device is None:
            if self.device.type == 'cuda':
                self._resource_device = torch.cuda.current_device()
            else:
                raise ValueError('CPU budget tests must specify a virtual resource_device.')
        self._stream = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None
        self._entries = OrderedDict()
        self._lock = RLock()
        self._in_use = False
        self._closed = False
        self.resident_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _drop(self, key):
        entry = self._entries[key]
        if entry.finished is not None:
            entry.finished.synchronize()
        # Release actual allocations before advertising reusable budget. Expired
        # lease mappings have already been revoked; clear the dictionary as well
        # so callbacks/stack frames retaining _Entry cannot pin device storage.
        entry.tensors.clear()
        if self.resources is not None and self.device.type == 'cuda':
            # Return now-unused allocator blocks to the driver before the shared
            # manager probes physical free memory. This flush can be expensive;
            # correctness takes precedence over cache throughput in this backend.
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        del self._entries[key]
        self.resident_bytes -= entry.size
        self.evictions += 1
        if entry.reservation is not None:
            entry.reservation.release()

    def _evict_shared(self, key, expected):
        # Admission callbacks must never wait on a cache lock: a cache miss may
        # own it while awaiting admission. Nonblocking failure lets the manager
        # try another resident instead of introducing lock-order inversion.
        if not self._lock.acquire(blocking=False):
            raise ResourceBusy('Expert cache is currently transferring or computing.')
        try:
            if self._entries.get(key) is expected:
                self._drop(key)
        finally:
            self._lock.release()

    def _copy(self, source, destination):
        if self._stream is None:
            for name, tensor in source.items():
                destination[name] = tensor.to(self.device, copy=True)
            return
        # Only the demanded expert is pinned. Track partially completed copies
        # in the entry itself so failure cleanup can free them before accounting.
        staging = {}
        try:
            staging = {name: tensor.contiguous().pin_memory() for name, tensor in source.items()}
            with torch.cuda.stream(self._stream):
                for name, tensor in staging.items():
                    destination[name] = tensor.to(self.device, non_blocking=True)
                ready = torch.cuda.Event()
                ready.record(self._stream)
            torch.cuda.current_stream(self.device).wait_event(ready)
            ready.synchronize()
        finally:
            self._stream.synchronize()
            staging.clear()

    @contextmanager
    def use(self, key):
        with self._lock, torch.inference_mode():
            if self._closed:
                raise RuntimeError('Expert cache is closed.')
            if self._in_use:
                raise RuntimeError('Nested expert leases are unsupported; finish one expert before acquiring another.')
            source = self.bank.get(key)
            size = tensor_bytes(source)
            if size > self.capacity_bytes:
                raise MemoryError(f'Expert requires {size} bytes, exceeding cache budget {self.capacity_bytes}.')
            fresh = key not in self._entries
            if not fresh:
                self.hits += 1
                entry = self._entries.pop(key)
                self._entries[key] = entry
            else:
                self.misses += 1
                while self.resident_bytes + size > self.capacity_bytes:
                    self._drop(next(iter(self._entries)))
                entry = _Entry({}, size)
                if self.resources is not None:
                    self._next_entry += 1
                    entry.reservation = self.resources.reserve(
                        f'{self._owner}:{self._next_entry}', 'llm',
                        device_bytes={self._resource_device: size},
                        evict=lambda: self._evict_shared(key, entry))
                self._entries[key] = entry
                self.resident_bytes += size
            try:
                # Resource leases protect actual copies and use. After enqueuing
                # CUDA work, eviction remains safe because _drop waits its event.
                with entry.reservation.lease() if entry.reservation is not None else nullcontext():
                    if fresh:
                        self._copy(source, entry.tensors)
                    if entry.finished is not None:
                        torch.cuda.current_stream(self.device).wait_event(entry.finished)
                    self._in_use = True
                    lease = _Lease(entry.tensors)
                    try:
                        yield lease
                    finally:
                        lease._tensors = None
                        if self._stream is not None:
                            entry.finished = torch.cuda.Event()
                            entry.finished.record(torch.cuda.current_stream(self.device))
                        self._in_use = False
            except BaseException:
                # In particular, partial transfer failures must not leave a
                # phantom reservation or tensor outside the tracked dictionary.
                if fresh and self._entries.get(key) is entry:
                    self._drop(key)
                raise

    def clear(self):
        """Free cache storage after consumers finish; retain host banks for reload."""
        with self._lock:
            if self._in_use:
                raise RuntimeError('Cannot clear an active expert lease.')
            while self._entries:
                self._drop(next(iter(self._entries)))

    def close(self):
        with self._lock:
            self.clear()
            self._closed = True
