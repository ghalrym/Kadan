"""Kadan's cross-workload RAM/VRAM admission and resident ownership.

Reservations are accounting, not allocations. Owners must free actual tensors in
an eviction callback or before release(). GPU budgets are independent; two 24 GiB
cards never imply a single 48 GiB allocation. Only inactive, evictable residents
may be removed. Workload adapters use a lease around generation and an exclusive
lease when they need other GPU residents fully unloaded.
"""
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path, PurePosixPath
import re
import threading
from typing import Callable, Literal

Workload = Literal['llm', 'image', 'video', 'speech', 'tts', 'decision']
WORKLOADS = frozenset(('llm', 'image', 'video', 'speech', 'tts', 'decision'))


class ResourceBusy(RuntimeError):
    pass


class ResourceExhausted(RuntimeError):
    pass


class ResourcePending(ResourceExhausted):
    """A supported reservation is temporarily blocked; its FIFO job may wait."""


class ResourceCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class MemoryCapacity:
    host_bytes: int
    device_bytes: dict[int, int] = field(default_factory=dict)


def _read_text(path: str | Path) -> str:
    """Read a kernel accounting file as text without trimming it; file errors propagate."""
    return Path(path).read_text()


def _cgroup_remaining() -> int | None:
    """Minimum visible ancestor hard-limit headroom for this Linux process.

    Resolve membership through mountinfo rather than assuming /sys/fs/cgroup.
    A cgroup namespace can report membership relative to the mounted root.
    Missing cgroup support falls back to host memory; a known finite limit whose
    usage cannot be read fails closed (zero headroom). No swap budget is counted.
    """
    try:
        memberships = []
        for line in _read_text('/proc/self/cgroup').splitlines():
            _, controllers, member = line.split(':', 2)
            if not controllers or 'memory' in controllers.split(','):
                memberships.append(('cgroup2' if not controllers else 'cgroup', member))
        mounts = _read_text('/proc/self/mountinfo').splitlines()
    except (OSError, ValueError):
        return None
    remaining = []
    for line in mounts:
        try:
            before, after = line.split(' - ', 1)
            fields, filesystem = before.split(), after.split()
            kind = filesystem[0]
            if kind not in ('cgroup', 'cgroup2'):
                continue
            if kind == 'cgroup' and 'memory' not in filesystem[2].split(','):
                continue
            # Kernel mountinfo escapes spaces, tabs, newlines, and backslashes.
            unescape = lambda value: re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), value)
            root = PurePosixPath(unescape(fields[3]))
            mount = Path(unescape(fields[4]))
            for membership_kind, member in memberships:
                if membership_kind != kind:
                    continue
                member_path = PurePosixPath(member)
                if not member_path.is_absolute() or '..' in member_path.parts:
                    continue
                try:
                    relative = member_path.relative_to(root)
                except ValueError:
                    # Membership is relative to a cgroup namespace root.
                    relative = member_path.relative_to('/')
                current = mount.joinpath(*relative.parts)
                while True:
                    limit_file = 'memory.max' if kind == 'cgroup2' else 'memory.limit_in_bytes'
                    usage_file = 'memory.current' if kind == 'cgroup2' else 'memory.usage_in_bytes'
                    try:
                        value = _read_text(current / limit_file).strip()
                        limit = None if value == 'max' else int(value)
                    except (OSError, ValueError):
                        limit = None
                    # v1 represents unlimited as PAGE_COUNTER_MAX (~2**63).
                    if limit is not None and limit >= 0 and (kind == 'cgroup2' or limit < 2**60):
                        try:
                            usage = int(_read_text(current / usage_file).strip())
                            remaining.append(max(0, limit - max(0, usage)))
                        except (OSError, ValueError):
                            remaining.append(0)
                    if current == mount:
                        break
                    current = current.parent
        except (ValueError, IndexError):
            continue
    return min(remaining) if remaining else None


def probe_memory() -> MemoryCapacity:
    """Available host/CUDA bytes, capped by visible cgroup RAM headroom.

    This is a conservative admission snapshot, not an allocation guarantee:
    other processes can allocate after probing. Importing this module never
    imports torch, and absent CUDA never produces a fabricated GPU budget.
    """
    host = None
    try:
        for line in _read_text('/proc/meminfo').splitlines():
            if line.startswith('MemAvailable:'):
                host = max(0, int(line.split()[1]) * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass
    if host is None:
        host = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    remaining = _cgroup_remaining()
    if remaining is not None:
        host = min(host, remaining)
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
    offload_on_handoff: bool = True
    active: int = 0
    evicting: bool = False


class Reservation:
    """An owner-specific handle; stale handles cannot release a replacement."""
    def __init__(self, manager: 'ResourceManager', owner: str, token: object):
        """Bind a handle to one manager owner and generation token; stale handles cannot release
        later reservations.
        """
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
                self._manager._changed()


class ResourceManager:
    def __init__(self, host_bytes: int, device_bytes: dict[int, int],
                 probe: Callable[[], MemoryCapacity] | None = None):
        """Create independent host and per-GPU byte budgets with an optional current-availability
        probe; this allocates no tensors.
        """
        self._validate(host_bytes, device_bytes)
        self.capacity = MemoryCapacity(host_bytes, dict(device_bytes))
        self._probe = probe
        self._lock = threading.RLock()
        self._revision = 0
        self._listeners = set()
        # Serialize admission/eviction transactions; callbacks run without the
        # state lock so release() and snapshots are safe from eviction callbacks.
        self._admission = threading.Lock()
        self._context_gate = threading.Lock()
        self._framework_contexts = {}
        self._residents: dict[str, _Resident] = {}
        self._exclusive: str | None = None
        self._used_host = 0
        self._used_devices: dict[int, int] = {}

    @property
    def revision(self):
        with self._lock:
            return self._revision

    def _changed(self):
        # Callers hold the state lock. Listeners only schedule an event-loop wake;
        # they never run admission callbacks or acquire another ownership lock.
        self._revision += 1
        for notify in tuple(self._listeners):
            try:
                notify()
            except RuntimeError:
                # A shutting-down event loop cannot invalidate physical cleanup.
                self._listeners.discard(notify)

    async def wait_for_change(self, revision, timeout=2.0):
        """Sleep on ownership changes; bounded reprobes cover external allocators.

        Register and compare under one lock so release cannot be lost between
        observing a revision and going to sleep. Cancellation removes the waiter.
        """
        loop = asyncio.get_running_loop()
        event = asyncio.Event()
        def notify():
            loop.call_soon_threadsafe(event.set)
        with self._lock:
            if self._revision != revision:
                return
            self._listeners.add(notify)
        try:
            try:
                await asyncio.wait_for(event.wait(), timeout)
            except TimeoutError:
                pass
        finally:
            with self._lock:
                self._listeners.discard(notify)

    @staticmethod
    def _validate(host: int, devices: dict[int, int]):
        """Reject negative, noninteger byte budgets and invalid GPU indices with ValueError."""
        if type(host) is not int or host < 0:
            raise ValueError('Host budget must be a nonnegative integer')
        if any(type(device) is not int or device < 0 or type(size) is not int or size < 0
               for device, size in devices.items()):
            raise ValueError('GPU IDs and budgets must be nonnegative integers')

    @staticmethod
    def _cancelled(event):
        """Raise ResourceCancelled if an optional cooperative cancellation event is set."""
        if event is not None and event.is_set():
            raise ResourceCancelled('Resource acquisition cancelled')

    @contextmanager
    def _transaction(self, event):
        # Cancellation is checked while waiting, rather than blocking forever
        # behind a potentially slow device-release callback.
        """Serialize admission and eviction, checking cancellation while waiting and always
        releasing the admission lock.
        """
        while not self._admission.acquire(timeout=.05):
            self._cancelled(event)
        try:
            self._cancelled(event)
            yield
        finally:
            self._admission.release()

    def _fits(self, host, devices):
        """Check logical capacity against current reservations; callers hold the state lock."""
        return (self._used_host + host <= self.capacity.host_bytes
                and all(self._used_devices.get(device, 0) + size
                        <= self.capacity.device_bytes[device] for device, size in devices.items()))

    def _record(self, owner, workload, host, devices, evict, offload_on_handoff):
        """Record admitted bytes under the state lock; cached totals avoid quadratic scans."""
        token = object()
        self._residents[owner] = _Resident(workload, host, devices, evict, token, offload_on_handoff)
        self._used_host += host
        for device, size in devices.items():
            self._used_devices[device] = self._used_devices.get(device, 0) + size
        self._changed()
        return Reservation(self, owner, token)

    def _remove(self, owner):
        """Remove exactly one resident under the state lock, including callback releases."""
        state = self._residents.pop(owner)
        self._used_host -= state.host_bytes
        for device, size in state.device_bytes.items():
            self._used_devices[device] -= size
        self._changed()

    def _physical_fits(self, host, devices):
        """Check fresh probe headroom, or accept when no probe was supplied; callers hold the state lock."""
        if self._probe is None:
            return True
        available = self._probe()
        return host <= available.host_bytes and all(
            size <= available.device_bytes.get(device, 0) for device, size in devices.items()
        )

    def _evict(self, owner):
        """Invoke an inactive owner cleanup outside the state lock. Remove accounting only after
        success; busy or failed cleanup preserves ownership.
        """
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
                self._remove(owner)

    def reserve(self, owner: str, workload: Workload, host_bytes: int = 0,
                device_bytes: dict[int, int] | None = None,
                evict: Callable[[], None] | None = None,
                cancel_event: threading.Event | None = None,
                offload_on_handoff: bool = True,
                allow_eviction: bool = True) -> Reservation:
        """Return an ownership handle after logical and physical admission. Evict eligible idle
        owners if needed; reject busy owners, cancellation or insufficient capacity. Callers
        allocate only afterward and provide callbacks that free actual tensors.
        offload_on_handoff=False is for bounded framework contexts, not model
        weights. Such contexts still count against GPU capacity and remain
        pressure-evictable through their cleanup callback.
        allow_eviction=False admits validation scratch only from free capacity;
        invalid incoming requests must not evict existing model residents.
        """
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
                if self._fits(host_bytes, devices) and self._physical_fits(host_bytes, devices):
                    self._cancelled(cancel_event)
                    return self._record(owner, workload, host_bytes, devices, evict, offload_on_handoff)
                if not allow_eviction:
                    raise ResourcePending('Insufficient free memory for non-disruptive validation')
                candidates = sorted(
                    [key for key, state in self._residents.items() if not state.active and state.evict is not None],
                    key=lambda key: (bool(self._residents[key].host_bytes),
                                     sum(self._residents[key].device_bytes.values())))
            for candidate in candidates:
                with self._lock:
                    available = self._probe() if self._probe else None
                    host_short = (self._used_host + host_bytes
                                  > self.capacity.host_bytes
                                  or (available is not None and host_bytes > available.host_bytes))
                    short_devices = {
                        device for device, size in devices.items()
                        if self._used_devices.get(device, 0) + size
                        > self.capacity.device_bytes[device]
                        or (available is not None and size > available.device_bytes.get(device, 0))
                    }
                    if not host_short and not short_devices:
                        break
                    state = self._residents.get(candidate)
                    if state is None or not (
                        (host_short and state.host_bytes) or any(state.device_bytes.get(device, 0) for device in short_devices)
                    ):
                        continue
                self._cancelled(cancel_event)
                try:
                    self._evict(candidate)
                except ResourceBusy:
                    # A generation/construction can become active after the
                    # candidate snapshot; never force it, try another resident.
                    continue
            with self._lock:
                if not self._fits(host_bytes, devices):
                    raise ResourcePending('Shared admission is waiting for resident cleanup')
                self._cancelled(cancel_event)
                if not self._physical_fits(host_bytes, devices):
                    raise ResourcePending('Shared admission is waiting for physical memory')
                return self._record(owner, workload, host_bytes, devices, evict, offload_on_handoff)

    def _release(self, owner, token):
        """Remove accounting for the matching owner token only; reject release while its leases
        remain active.
        """
        with self._lock:
            state = self._residents.get(owner)
            if state is None or state.token is not token:
                return
            if state.active:
                raise ResourceBusy('Cannot release a workload with active leases')
            self._remove(owner)

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
                self._changed()

    def offload_inactive_devices(self, workload: Workload, cancel_event=None):
        """Offload through existing callbacks; preserve host banks and active leases."""
        with self._transaction(cancel_event):
            with self._lock:
                if self._exclusive is not None:
                    raise ResourceBusy('Another workload holds exclusive GPU access')
                victims = [key for key, state in self._residents.items()
                           if state.workload != workload and state.offload_on_handoff and any(state.device_bytes.values())]
                if any(self._residents[key].active or self._residents[key].evict is None for key in victims):
                    raise ResourceBusy('Other GPU workloads are active or not evictable')
            for victim in victims:
                self._cancelled(cancel_event)
                self._evict(victim)

    def offload_workload_devices(self, workload: Workload, cancel_event=None):
        """Park one wrapper's device allocations using the same admission ownership."""
        with self._transaction(cancel_event):
            with self._lock:
                if self._exclusive is not None:
                    raise ResourceBusy('Another workload holds exclusive GPU access')
                victims = [key for key, state in self._residents.items()
                           if state.workload == workload and state.offload_on_handoff and any(state.device_bytes.values())]
                if any(self._residents[key].active or self._residents[key].evict is None for key in victims):
                    raise ResourceBusy('GPU workload is active or not evictable')
            for victim in victims:
                self._cancelled(cancel_event)
                self._evict(victim)

    def framework_context(self, device, size):
        """Admit a process-lifetime Torch CUDA context before image CUDA execution.

        Parking tensors or empty_cache cannot destroy a shared Python context.
        This non-evictable envelope stays charged until this process exits; native
        child contexts have their own separately reaped ownership instead.
        """
        with self._context_gate:
            if device not in self._framework_contexts:
                self._framework_contexts[device] = (size, self.reserve(
                    f'framework-context:{device}', 'image', device_bytes={device: size},
                    offload_on_handoff=False))
            admitted, handle = self._framework_contexts[device]
            if size > admitted:
                raise ResourceExhausted('Framework context exceeds its admitted envelope')
            return handle

    def available_devices(self, *, reclaim=False):
        """Snapshot execution headroom without evicting or pooling device capacities.

        Reclaimable accounting is only a planning upper bound: every subsequent
        reserve still probes physical memory and performs real callback cleanup.
        Active/non-evictable residents are never credited as reclaimable.
        """
        with self._lock:
            physical = self._probe().device_bytes if self._probe else self.capacity.device_bytes
            credit = {}
            if reclaim:
                for state in self._residents.values():
                    if not state.active and not state.evicting and state.evict is not None:
                        for i, size in state.device_bytes.items():
                            credit[i] = credit.get(i, 0) + size
            return {i: max(0, min(capacity - self._used_devices.get(i, 0) + credit.get(i, 0),
                                 physical.get(i, 0) + credit.get(i, 0)))
                    for i, capacity in self.capacity.device_bytes.items()}

    def snapshot(self) -> dict:
        """Return a lock-consistent copy of budgets and reservation metadata, not a measurement of
        allocated tensors.
        """
        with self._lock:
            return {
                'host_capacity_bytes': self.capacity.host_bytes,
                'device_capacity_bytes': dict(self.capacity.device_bytes),
                'exclusive_owner': self._exclusive,
                'reservations': {
                    owner: {'workload': state.workload, 'host_bytes': state.host_bytes,
                            'device_bytes': dict(state.device_bytes), 'active_leases': state.active,
                            'evicting': state.evicting, 'offload_on_handoff': state.offload_on_handoff}
                    for owner, state in self._residents.items()
                },
            }
