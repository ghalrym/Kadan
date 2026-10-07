"""Idle host-bank eviction and on-demand reconstruction for native adapters.

Only metadata survives host eviction. This wrapper shares the existing resource
manager; it does not introduce an engine, worker, or separate memory accountant.
"""
import threading

from .conversation import ConversationCache

from api.inference.resources import ResourceBusy, ResourceCancelled


def _check_cancel(event):
    """Raise ResourceCancelled when the optional cancellation event requests that construction or
    waiting stop.
    """
    if event is not None and event.is_set():
        raise ResourceCancelled('Model reconstruction cancelled')


class _HostEvictionResources:
    """Attach whole-adapter cleanup to its host reservations."""
    def __init__(self, manager, on_host_evict):
        """Retain the shared manager and whole-model cleanup callback; create no separate memory budget."""
        self._manager = manager
        self._on_host_evict = on_host_evict

    def reserve(self, owner, workload, host_bytes=0, device_bytes=None,
                evict=None, cancel_event=None):
        """Delegate admission, attaching whole-model cleanup only to host-only reservations;
        preserve callbacks for device and request allocations.
        """
        return self._manager.reserve(owner, workload, host_bytes=host_bytes,
            device_bytes=device_bytes, evict=self._on_host_evict if host_bytes and not device_bytes and evict is None else evict,
            cancel_event=cancel_event)

    def __getattr__(self, name):
        """Forward other resource-manager operations to the same shared instance."""
        return getattr(self._manager, name)


class ReloadableAdapter:
    """Serialize adapter ownership; reject eviction while allocation/work is active.

    The gate deliberately is NOT reentrant: resource admission can synchronously
    call eviction on this same thread while a constructor is still allocating.
    Nonblocking eviction avoids an admission-lock/adapter-lock inversion.
    """
    def __init__(self, factory, entry, path, resources, device='cuda:0', cancel_event=None):
        """Construct a native adapter under a nonreentrant ownership gate, retaining checkpoint
        metadata for later idle reloads.
        """
        self._factory, self._entry, self._path, self._device = factory, entry, path, device
        self._gate = threading.Lock()
        self._closed = False
        self._inner = None
        self._context_was_configured = False
        self.configured_context_limit = None
        self.effective_context_limit = None
        self.supported_context_limit = None
        self.conversations = ConversationCache(resources)
        self._resources = _HostEvictionResources(resources, self._evict_host)
        with self._gate:
            self._construct(cancel_event)

    def _construct(self, cancel_event):
        """Build while the caller holds the ownership gate, then restore saved context metadata.
        Native constructors own partial-allocation rollback.
        """
        _check_cancel(cancel_event)
        # Native constructors own rollback of partially allocated tensors and
        # reservations. Publish only a completely constructed adapter.
        inner = self._factory(self._entry, self._path, self._resources,
                              device=self._device, cancel_event=cancel_event)
        self._inner = inner
        try:
            if self._context_was_configured:
                from api.inference.llm.context import configure_context
                configure_context(inner, self.configured_context_limit)
        except BaseException:
            inner.close()
            self._inner = None
            raise
        self._remember_context(inner)

    def _remember_context(self, inner):
        """Copy context limits as scalar metadata so they survive release of model tensors."""
        for name in ('configured_context_limit', 'effective_context_limit', 'supported_context_limit'):
            setattr(self, name, getattr(inner, name, None))

    def configure_context(self, value):
        """Validate and remember a limit without forcing an evicted model to reload; reject
        explicit closure and unsupported limits.
        """
        from api.inference.llm.context import configure_context
        with self._gate:
            if self._closed:
                raise RuntimeError('Model has been explicitly closed')
            if self._inner is None:
                # Configuration normally occurs just after load. Validate against
                # the retained architectural maximum if already host-evicted.
                if value is not None and (type(value) is not int or value < 1 or
                        self.supported_context_limit is None or value > self.supported_context_limit):
                    raise ValueError('Configured context exceeds the supported model limit')
                self.configured_context_limit = value
                self.effective_context_limit = value or self.supported_context_limit
            else:
                configure_context(self._inner, value)
                self._remember_context(self._inner)
            self.conversations.clear()
            self._context_was_configured = True

    @property
    def is_resident(self):
        """Report whether the current inner adapter is GPU-resident; reading status does not
        trigger restoration.
        """
        inner = self._inner
        return not self._closed and inner is not None and inner.is_resident

    def _evict_host(self):
        """Nonblockingly claim idle ownership, close the inner adapter, and retain only metadata.
        Busy work rejects eviction. Cleanup failure retains the inner handle for
        retry, although that adapter may already have released some state.
        """
        if not self._gate.acquire(blocking=False):
            raise ResourceBusy('Model is constructing, generating or closing; host memory is in use')
        try:
            self.conversations.clear()
            if self._inner is not None:
                # close() frees actual banks/caches/model state and synchronizes
                # device use before releasing its resource reservations. Preserve
                # the handle on failure so a subsequent close can retry cleanup.
                self._inner.close()
                self._inner = None
        finally:
            self._gate.release()

    def generate(self, messages, max_new_tokens=256, cancel_event=None, on_event=None, conversation_id=None):
        """Serialize generation, rebuilding an idle-evicted adapter when needed. Check cancellation
        while waiting; explicitly closed adapters never reload.
        """
        while not self._gate.acquire(timeout=.05):
            _check_cancel(cancel_event)
        try:
            _check_cancel(cancel_event)
            if self._closed:
                raise RuntimeError('Model has been explicitly closed')
            if self._inner is None:
                self._construct(cancel_event)
            return self._inner.generate(messages, max_new_tokens=max_new_tokens,
                                        cancel_event=cancel_event, **({"on_event": on_event} if on_event else {}),
                                        **({"conversation_cache": self.conversations, "conversation_id": conversation_id}
                                           if conversation_id else {}))
        except BaseException:
            if conversation_id:
                self.conversations.invalidate(conversation_id)
            raise
        finally:
            self._gate.release()

    def close(self):
        """Wait for ownership, release the inner adapter and permanently close this wrapper;
        preserve the inner handle if cleanup fails.
        """
        with self._gate:
            self.conversations.clear()
            if self._inner is not None:
                self._inner.close()
                self._inner = None
            self._closed = True
