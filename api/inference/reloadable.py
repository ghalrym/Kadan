"""Idle host-bank eviction and on-demand reconstruction for native adapters.

Only metadata survives host eviction. This wrapper shares the existing resource
manager; it does not introduce an engine, worker, or separate memory accountant.
"""
import threading

from api.inference.resources import ResourceBusy, ResourceCancelled


def _check_cancel(event):
    if event is not None and event.is_set():
        raise ResourceCancelled('Model reconstruction cancelled')


class _HostEvictionResources:
    """Attach whole-adapter cleanup to its host reservations."""
    def __init__(self, manager, on_host_evict):
        self._manager = manager
        self._on_host_evict = on_host_evict

    def reserve(self, owner, workload, host_bytes=0, device_bytes=None,
                evict=None, cancel_event=None):
        return self._manager.reserve(owner, workload, host_bytes=host_bytes,
            device_bytes=device_bytes, evict=self._on_host_evict if host_bytes else evict,
            cancel_event=cancel_event)

    def __getattr__(self, name):
        return getattr(self._manager, name)


class ReloadableAdapter:
    """Serialize adapter ownership; reject eviction while allocation/work is active.

    The gate deliberately is NOT reentrant: resource admission can synchronously
    call eviction on this same thread while a constructor is still allocating.
    Nonblocking eviction avoids an admission-lock/adapter-lock inversion.
    """
    def __init__(self, factory, entry, path, resources, device='cuda:0', cancel_event=None):
        self._factory, self._entry, self._path, self._device = factory, entry, path, device
        self._gate = threading.Lock()
        self._closed = False
        self._inner = None
        self._resources = _HostEvictionResources(resources, self._evict_host)
        with self._gate:
            self._construct(cancel_event)

    def _construct(self, cancel_event):
        _check_cancel(cancel_event)
        # Native constructors own rollback of partially allocated tensors and
        # reservations. Publish only a completely constructed adapter.
        inner = self._factory(self._entry, self._path, self._resources,
                              device=self._device, cancel_event=cancel_event)
        self._inner = inner

    @property
    def is_resident(self):
        inner = self._inner
        return not self._closed and inner is not None and inner.is_resident

    def _evict_host(self):
        if not self._gate.acquire(blocking=False):
            raise ResourceBusy('Model is constructing, generating or closing; host memory is in use')
        try:
            if self._inner is not None:
                # close() frees actual banks/caches/model state and synchronizes
                # device use before releasing its resource reservations. Preserve
                # the handle on failure so a subsequent close can retry cleanup.
                self._inner.close()
                self._inner = None
        finally:
            self._gate.release()

    def generate(self, messages, max_new_tokens=256, cancel_event=None):
        while not self._gate.acquire(timeout=.05):
            _check_cancel(cancel_event)
        try:
            _check_cancel(cancel_event)
            if self._closed:
                raise RuntimeError('Model has been explicitly closed')
            if self._inner is None:
                self._construct(cancel_event)
            return self._inner.generate(messages, max_new_tokens=max_new_tokens,
                                        cancel_event=cancel_event)
        finally:
            self._gate.release()

    def close(self):
        with self._gate:
            if self._inner is not None:
                self._inner.close()
                self._inner = None
            self._closed = True
