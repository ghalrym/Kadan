"""Protocol-v2 leaf under Python's global FIFO and parent memory admission.

The child exclusively owns its CUDA context. Never host unrelated CUDA work in
that process: its ledger cannot discover allocations made outside its ownership.
"""
from contextlib import contextmanager
import re

from api.inference.llm.context import ContextLimitError, ContextMemoryError, resolve_context
from api.inference.llm.native import (
    HEADROOM_BYTES, METADATA_BYTES, MIB, NATIVE_HOST_BYTES, PYTHON_HOST_BYTES,
    NativeAdapter, NativeProtocolError, WorkerProcess, numbers,
)
from api.inference.placement import select_device
from api.inference.resources import ResourceBusy, ResourceExhausted

RESIDENT_HOST_BYTES = NATIVE_HOST_BYTES + 256 * MIB
CACHE_RAM_BYTES = 256 * MIB
# Bound registered source references, not total on-disk checkpoint storage.
CACHE_COLD_BYTES = 64 * 1024**3


class ResidentAdapter(NativeAdapter):
    def __init__(self, *args, cache_ram_bytes=None, **kwargs):
        super().__init__(*args, **kwargs)
        if cache_ram_bytes is not None and (type(cache_ram_bytes) is not int or not 0 <= cache_ram_bytes <= CACHE_COLD_BYTES):
            raise ValueError('Invalid native RAM cache limit')
        self.requested_cache_bytes = cache_ram_bytes
        self.cache_capacity = 0
        self.packed_weight_bytes = 0
        self.cache_statistics = None
        self.backing = None
        self.context = None
        self.session = None
        self.request_id = self.last_id = 0
        self.parked = False
        self.quarantined = False

    def completion_timeout(self, generation_timeout):
        """Bound a completion including a possible pressure-evicted child rebuild.

        Planning, readiness/loading, and start/restore each have a load deadline.
        Budget all three even for a currently warm child: pressure eviction may
        happen before generate acquires its ownership lock. Token IPC keeps its
        shorter step deadline. This is a combined outer bound, not three retries.
        """
        return generation_timeout + 3 * self.load_timeout

    @property
    def is_resident(self):
        return self.worker_alive and not self.quarantined and not self.parked and self.reservation is not None

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
        self.supported_context_limit, self.effective_context_limit = supported, effective
        self.capacity = effective
        try:
            self.host = self.resources.reserve(self.owner + ':host', 'llm',
                host_bytes=RESIDENT_HOST_BYTES + PYTHON_HOST_BYTES,
                evict=self._evict, cancel_event=self.cancel)
            with self.host.lease(self.cancel):
                self.worker = WorkerProcess()
                self.worker.start([str(self.binary), '--plan-resident', str(self.root), str(effective), str(METADATA_BYTES)])
                host, arena, vocab, capacity, staging, packed, tensors = numbers(
                    self.worker.read(self.load_timeout, self.cancel), ['plan', '3'], 7)
                self.worker.finish()
                self.worker.stop()
                self.worker = None
                if host != RESIDENT_HOST_BYTES or capacity != effective or not 0 < vocab <= 262144 or arena <= 0 or not 0 < staging <= MIB:
                    raise NativeProtocolError('Native resident plan violates adapter bounds')
                if not 0 < packed <= CACHE_COLD_BYTES or not 0 < tensors <= 131072:
                    raise NativeProtocolError('Packed text model exceeds cache metadata/source bounds')
                self.packed_weight_bytes = packed
                self.packed_tensor_count = tensors
                remaining = max(0, self.resources.capacity.host_bytes - RESIDENT_HOST_BYTES - PYTHON_HOST_BYTES)
                preferred = packed if self.requested_cache_bytes is None else min(packed, self.requested_cache_bytes)
                self.cache_capacity = preferred if preferred <= remaining else min(CACHE_RAM_BYTES, remaining)
                self.backing = self.resources.reserve(self.owner + ':backing', 'llm',
                    host_bytes=self.cache_capacity, evict=self._evict, cancel_event=self.cancel)
                with self.backing.lease(self.cancel):
                    self.vocabulary, self.arena = vocab, arena
                    self.device = select_device(self.resources, arena + HEADROOM_BYTES, self.device)
                    index = int(self.device[5:])
                    self.context = self.resources.reserve(self.owner + ':context', 'llm',
                        device_bytes={index: HEADROOM_BYTES}, evict=self._evict,
                        offload_on_handoff=False, cancel_event=self.cancel)
                    with self.context.lease(self.cancel):
                        self._reserve_arena(self.cancel)
                        with self.reservation.lease(self.cancel):
                            self.tokenizer = self.tokenizer_factory(self.root)
                            self.worker = WorkerProcess()
                            self.worker.start([str(self.binary), '--serve-resident', str(self.root), str(index),
                                str(effective), str(host + self.cache_capacity), str(arena + HEADROOM_BYTES),
                                str(HEADROOM_BYTES), str(self.cache_capacity), str(CACHE_COLD_BYTES)])
                            ready = self.worker.read(self.load_timeout, self.cancel).split()
                            if len(ready) != 5 or ready[:2] != ['ready', '2'] or not re.fullmatch('[0-9a-f]{32}', ready[2]):
                                raise NativeProtocolError('Invalid native resident session')
                            if numbers(' '.join(ready[3:]), [], 2) != [vocab, effective]:
                                raise NativeProtocolError('Native readiness differs from admitted plan')
                            self.session = ready[2]
                            self.cache_statistics = None
                            self._read_cache_stats(self.cancel)
                            self.last_id = self.request_id = 0
                            self.parked = False
        except BaseException:
            self._close_locked()
            raise

    def _reserve_arena(self, cancel):
        self.reservation = self.resources.reserve(self.owner + ':device', 'llm',
            device_bytes={int(self.device[5:]): self.arena}, evict=self.offload_to_ram, cancel_event=cancel)

    @contextmanager
    def _leases(self, cancel):
        with self.host.lease(cancel), self.backing.lease(cancel), self.context.lease(cancel), self.reservation.lease(cancel):
            yield

    def _restore_locked(self, cancel):
        if self.quarantined:
            raise NativeProtocolError('Native cleanup is uncertain; close before reuse')
        if self.parked and self.worker_alive:
            # Protect backing/context from pressure eviction during admission.
            with self.host.lease(cancel), self.backing.lease(cancel), self.context.lease(cancel):
                try:
                    self._reserve_arena(cancel)
                except (ResourceBusy, ResourceExhausted) as exc:
                    # No command or physical allocation has occurred; safely retry.
                    raise ContextMemoryError(str(exc)) from exc
            self.parked = False

    def _reply(self, command, reply, request_id, cancel=None, count=0, timeout=None):
        return numbers(self.worker.exchange(f'{command} {self.session} {request_id}',
            self.step_timeout if timeout is None else timeout, cancel), [reply, self.session, str(request_id)], count)

    def _read_cache_stats(self, cancel=None):
        fields = ('capacity_bytes', 'retained_bytes', 'registered_bytes', 'hits', 'misses',
                  'hit_bytes', 'source_bytes', 'evictions', 'entries')
        values = self._reply('cache', 'cache', 0, cancel, count=len(fields))
        stats = dict(zip(fields, values))
        if (stats['capacity_bytes'] != self.cache_capacity or
                stats['retained_bytes'] > self.cache_capacity or
                stats['retained_bytes'] > stats['registered_bytes'] or
                stats['registered_bytes'] > self.packed_weight_bytes or stats['entries'] > self.packed_tensor_count):
            raise NativeProtocolError('Native cache telemetry violates admitted bounds')
        self.cache_statistics = stats
        return dict(stats)

    def cache_stats(self):
        """Last acknowledged idle snapshot; counters measure application reads, not SSD IO."""
        with self._lock:
            return None if self.cache_statistics is None else dict(self.cache_statistics)

    def _begin_request(self, cancel):
        ids = numbers(self.worker.exchange(f'submit {self.session} 0', self.step_timeout, cancel),
            ['queued', self.session], 1)
        if not self.last_id < ids[0] < 2**64:
            raise NativeProtocolError('Native FIFO request ID was reused or invalid')
        self.request_id = self.last_id = ids[0]
        # Start can reload all packed weights after park; token deadlines do not apply.
        self._reply('start', 'started', self.request_id, cancel, timeout=self.load_timeout)

    def _end_request(self, cancel):
        self._reply('end', 'ended', self.request_id, cancel)
        self.request_id = 0
        self._read_cache_stats(cancel)

    def _step(self, token, stop, expected, cancel):
        selected, eos, progress = numbers(self.worker.exchange(
            f'step {self.session} {self.request_id} {token} {int(stop)}', self.step_timeout, cancel),
            ['token', self.session, str(self.request_id)], 3)
        if selected >= self.vocabulary or eos not in (0, 1) or progress != expected:
            raise NativeProtocolError('Native step violates vocabulary/EOS/progress contract')
        return selected, bool(eos)

    def offload_to_ram(self):
        if not self._lock.acquire(blocking=False):
            raise ResourceBusy('Native worker is active')
        try:
            if self.reservation is None:
                return
            if self.quarantined or self.request_id or not self.worker_alive:
                raise NativeProtocolError('Native worker cleanup is not confirmed')
            # Do not release the model envelope if the ack is missing or stale.
            self._reply('park', 'parked', 0)
            self._read_cache_stats()
            self.reservation.release()
            self.reservation = None
            self.parked = True
        except BaseException:
            self.quarantined = True
            raise
        finally:
            self._lock.release()

    def _close_command(self):
        return f'close {self.session} 0'

    def _close_reply(self):
        return f'closed {self.session} 0'

    def _close_locked(self):
        # Reap first. An unconfirmed stop leaves all parent envelopes charged.
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        self.tokenizer = None
        for name in ('reservation', 'context', 'backing', 'host'):
            handle = getattr(self, name)
            if handle is not None:
                handle.release()
                setattr(self, name, None)
        self.cache_statistics = None
        self.session = None
        self.request_id = 0
        self.parked = False
        self.quarantined = False


def build_resident(entry, path, resources, device='auto', cancel_event=None):
    return ResidentAdapter(entry, path, resources, device, cancel_event)
