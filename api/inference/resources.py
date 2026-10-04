"""Kadan's cross-workload RAM/VRAM admission and resident ownership.

Reservations are accounting, not allocations. Owners must free actual tensors in
an eviction callback or before release(). GPU budgets are independent; two 24 GiB
cards never imply a single 48 GiB allocation. Only inactive, evictable residents
may be removed. Workload adapters use a lease around generation and an exclusive
lease when they need other GPU residents fully unloaded.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import os
import threading
from typing import Callable, Literal

Workload = Literal['llm', 'image', 'video', 'speech', 'decision']
WORKLOADS = frozenset(('llm', 'image', 'video', 'speech', 'decision'))


class ResourceBusy(RuntimeError):
    pass


class ResourceExhausted(RuntimeError):
    pass


class ResourceCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class MemoryCapacity:
    host_bytes: int
    device_bytes: dict[int, int] = field(default_factory=dict)


def probe_memory() -> MemoryCapacity:
    """Available host/CUDA bytes; importing this module never imports torch.

    Linux MemAvailable includes reclaimable caches. CUDA absence is represented
    by an empty device map, not a fabricated GPU budget.
    """
    host = 0
    try:
        with open('/proc/meminfo') as stream:
            for line in stream:
                if line.startswith('MemAvailable:'):
                    host = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    if not host:
        host = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    devices = {}
    try:
        import torch
    except ImportError:
        return MemoryCapacity(host, devices)
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(device)
            devices[device] = int(free)
    return MemoryCapacity(host, devices)


@dataclass
class _Resident:
    workload: str
    host_bytes: int
    device_bytes: dict[int, int]
    evict: Callable[[], None] | None
    token: object
    active: int = 0
    evicting: bool = False


class Reservation:
    """An owner-specific handle; stale handles cannot release a replacement."""
    def __init__(self, manager: 'ResourceManager', owner: str, token: object):
        self._manager = manager
        self.owner = owner
        self._token = token

    def release(self):
        """Call after freeing actual allocations; safe to repeat after eviction."""
        self._manager._release(self.owner, self._token)

    @contextmanager
    def lease(self, cancel_event: threading.Event | None = None):
        """Protect residency during work, including on exceptions/cancellation."""
        with self._manager._lock:
            self._manager._cancelled(cancel_event)
            state = self._manager._residents.get(self.owner)
            if state is None or state.token is not self._token:
                raise ResourceBusy('Reservation is no longer resident; reload the workload')
            if state.evicting or self._manager._exclusive not in (None, self.owner):
                raise ResourceBusy('Another workload owns GPU access')
            state.active += 1
        try:
            yield self
        finally:
            with self._manager._lock:
                state.active -= 1


class ResourceManager:
    def __init__(self, host_bytes: int, device_bytes: dict[int, int],
                 probe: Callable[[], MemoryCapacity] | None = None):
        self._validate(host_bytes, device_bytes)
        self.capacity = MemoryCapacity(host_bytes, dict(device_bytes))
        self._probe = probe
        self._lock = threading.RLock()
        # Serialize admission/eviction transactions; callbacks run without the
        # state lock so release() and snapshots are safe from eviction callbacks.
        self._admission = threading.Lock()
        self._residents: dict[str, _Resident] = {}
        self._exclusive: str | None = None

    @staticmethod
    def _validate(host: int, devices: dict[int, int]):
        if type(host) is not int or host < 0:
            raise ValueError('Host budget must be a nonnegative integer')
        if any(type(device) is not int or device < 0 or type(size) is not int or size < 0
               for device, size in devices.items()):
            raise ValueError('GPU IDs and budgets must be nonnegative integers')

    @staticmethod
    def _cancelled(event):
        if event is not None and event.is_set():
            raise ResourceCancelled('Resource acquisition cancelled')

    @contextmanager
    def _transaction(self, event):
        # Cancellation is checked while waiting, rather than blocking forever
        # behind a potentially slow device-release callback.
        while not self._admission.acquire(timeout=.05):
            self._cancelled(event)
        try:
            self._cancelled(event)
            yield
        finally:
            self._admission.release()

    def _fits(self, host, devices):
        return (sum(item.host_bytes for item in self._residents.values()) + host <= self.capacity.host_bytes
                and all(sum(item.device_bytes.get(device, 0) for item in self._residents.values()) + size
                        <= self.capacity.device_bytes[device] for device, size in devices.items()))

    def _evict(self, owner):
        with self._lock:
            state = self._residents.get(owner)
            if state is None:
                return
            if state.active or state.evict is None or state.evicting:
                raise ResourceBusy(f'Workload {owner} cannot be evicted while active or without cleanup')
            state.evicting = True
        try:
            state.evict()
        except BaseException:
            with self._lock:
                state.evicting = False
            raise
        with self._lock:
            if self._residents.get(owner) is state:
                del self._residents[owner]

    def reserve(self, owner: str, workload: Workload, host_bytes: int = 0,
                device_bytes: dict[int, int] | None = None,
                evict: Callable[[], None] | None = None,
                cancel_event: threading.Event | None = None) -> Reservation:
        devices = dict(device_bytes or {})
        self._validate(host_bytes, devices)
        if not owner or workload not in WORKLOADS:
            raise ValueError('A nonempty owner and supported workload are required')
        if host_bytes > self.capacity.host_bytes or any(
            device not in self.capacity.device_bytes or size > self.capacity.device_bytes[device]
            for device, size in devices.items()
        ):
            raise ResourceExhausted('Request exceeds an individual RAM or GPU budget')
        with self._transaction(cancel_event):
            with self._lock:
                if owner in self._residents:
                    raise ResourceBusy('Owner already has a reservation; release before resizing')
                if self._exclusive not in (None, owner):
                    raise ResourceBusy('Another workload holds exclusive GPU access')
                candidates = [key for key, state in self._residents.items()
                              if not state.active and state.evict is not None]
            for candidate in candidates:
                with self._lock:
                    if self._fits(host_bytes, devices):
                        break
                self._cancelled(cancel_event)
                self._evict(candidate)
            with self._lock:
                if not self._fits(host_bytes, devices):
                    raise ResourceExhausted('Insufficient unreserved RAM/VRAM; active workloads cannot be evicted')
                self._cancelled(cancel_event)
                if self._probe:
                    available = self._probe()
                    if host_bytes > available.host_bytes or any(
                        size > available.device_bytes.get(device, 0) for device, size in devices.items()
                    ):
                        raise ResourceExhausted('Physical available memory is below the requested reservation')
                token = object()
                self._residents[owner] = _Resident(workload, host_bytes, devices, evict, token)
                return Reservation(self, owner, token)

    def _release(self, owner, token):
        with self._lock:
            state = self._residents.get(owner)
            if state is None or state.token is not token:
                return
            if state.active:
                raise ResourceBusy('Cannot release a workload with active leases')
            del self._residents[owner]

    @contextmanager
    def exclusive(self, owner: str, cancel_event: threading.Event | None = None):
        """Evict other inactive GPU owners, hold admission for this owner only.

        A future image/video adapter can enter this before reserve(). Existing
        active generations cause a conflict; this never forcibly frees tensors
        used by an in-flight request. Exiting restores availability, not models;
        evicted owners must reload on their next request.
        """
        if not owner:
            raise ValueError('A nonempty owner is required')
        with self._transaction(cancel_event):
            with self._lock:
                if self._exclusive is not None:
                    raise ResourceBusy('Exclusive workload already active')
                victims = [key for key, state in self._residents.items()
                           if key != owner and any(state.device_bytes.values())]
                if any(self._residents[key].active or self._residents[key].evict is None for key in victims):
                    raise ResourceBusy('Other GPU workloads are active or not evictable')
                # Prevent new generation leases during callback cleanup.
                self._exclusive = owner
            try:
                for victim in victims:
                    self._cancelled(cancel_event)
                    self._evict(victim)
                self._cancelled(cancel_event)
            except BaseException:
                with self._lock:
                    self._exclusive = None
                raise
        try:
            yield
        finally:
            with self._lock:
                self._exclusive = None

    def snapshot(self) -> dict:
        with self._lock:
            return {
                'host_capacity_bytes': self.capacity.host_bytes,
                'device_capacity_bytes': dict(self.capacity.device_bytes),
                'exclusive_owner': self._exclusive,
                'reservations': {
                    owner: {'workload': state.workload, 'host_bytes': state.host_bytes,
                            'device_bytes': dict(state.device_bytes), 'active_leases': state.active,
                            'evicting': state.evicting}
                    for owner, state in self._residents.items()
                },
            }
