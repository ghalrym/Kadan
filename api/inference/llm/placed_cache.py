"""In-process packed-weight placement; attention/KV remain on the primary device."""
from contextlib import contextmanager, ExitStack, nullcontext
from uuid import uuid4

import torch

from api.inference.placement import place_packed
from api.inference.resources import ResourceExhausted
from .offload import ExpertCache, tensor_bytes

GIB = 1024**3
REMOTE_SCRATCH = 128 * 1024**2


class PlacedExpertCache:
    def __init__(self, bank, capacity_bytes, device, *, resources, owner):
        self.bank, self.resources, self.owner = bank, resources, owner
        self.device = torch.device(device)
        self.capacity_bytes = capacity_bytes
        self.plan = None
        self.caches = {}
        self._closed = False
        self._warm = False

    def _plan(self):
        if self._closed:
            raise RuntimeError('Expert cache is closed')
        if self.plan is not None:
            return
        sizes = {key: tensor_bytes(value) for key, value in self.bank._experts.items()}
        available = self.resources.available_devices()
        headroom = {i: (2 * GIB if i == self.device.index else REMOTE_SCRATCH) for i in available}
        try:
            self.plan = place_packed(sizes, available, self.device.index, headroom)
        except ResourceExhausted:
            self.plan = place_packed(sizes, self.resources.available_devices(reclaim=True), self.device.index, headroom)
        self.caches = {i: ExpertCache(self.bank, capacity, f'cuda:{i}', resources=self.resources,
                                     owner=f'{self.owner}:gpu:{i}')
                       for i, capacity in self.plan.capacities.items()}

    @contextmanager
    def execution(self, cancel=None):
        self._plan()
        with ExitStack() as leases:
            try:
                # Each remote GPU executes one projection at a time. Hold its
                # decode/activation workspace independently of packed-weight bytes.
                for i in self.caches:
                    if i != self.device.index:
                        reservation = self.resources.reserve(f'{self.owner}:scratch:{uuid4().hex}', 'llm',
                            device_bytes={i: REMOTE_SCRATCH}, cancel_event=cancel)
                        leases.callback(reservation.release)
                        leases.enter_context(reservation.lease(cancel))
                if self.plan.fully_resident and not self._warm:
                    # Keep a request execution floor available while warming the
                    # whole bank. Larger requests still admit actual KV/workspace
                    # and may pressure-evict idle packed entries later.
                    floor = self.resources.reserve(f'{self.owner}:warm:{uuid4().hex}', 'llm',
                        device_bytes={self.device.index: 2 * GIB}, cancel_event=cancel)
                    try:
                        with floor.lease(cancel):
                            for key in self.plan.assignments:
                                self.resources._cancelled(cancel)
                                with self.use(key):
                                    pass
                        self._warm = True
                    finally:
                        floor.release()
                yield
            finally:
                for i in self.caches:
                    torch.cuda.synchronize(i)

    @contextmanager
    def use(self, key):
        self._plan()
        with self.caches[self.plan.assignments[key]].use(key) as tensors:
            yield tensors

    @property
    def resident_bytes(self):
        return sum(cache.resident_bytes for cache in self.caches.values())

    @property
    def hits(self):
        return sum(cache.hits for cache in self.caches.values())

    @property
    def misses(self):
        return sum(cache.misses for cache in self.caches.values())

    @property
    def evictions(self):
        return sum(cache.evictions for cache in self.caches.values())

    def clear(self):
        for cache in self.caches.values():
            cache.close()
        self.caches.clear()
        self.plan = None
        self._warm = False

    def close(self):
        self.clear()
        self._closed = True


def make_cache(bank, capacity_bytes, device, *, resources=None, owner=None):
    if torch.device(device).type == 'cpu':
        return ExpertCache(bank, capacity_bytes, device=device)
    return PlacedExpertCache(bank, capacity_bytes, device, resources=resources, owner=owner)


def cache_execution(cache, cancel):
    return cache.execution(cancel) if isinstance(cache, PlacedExpertCache) else nullcontext()
