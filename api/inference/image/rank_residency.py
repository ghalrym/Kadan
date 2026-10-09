"""Two-rank lifecycle under the sole global FIFO.

The existing Redis consumer is the only request queue. The transport
must supervise both children and attest physical cleanup before returning True
from stop(). This controller never infers cleanup from EOF or a single rank.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import re
import math
import logging
import threading
import time
from typing import Protocol
from uuid import uuid4

from api.inference.resources import ResourceBusy, ResourceCancelled, ResourceRecoveryRequired


@dataclass(frozen=True)
class ImageRankBudget:
    host_bytes: int
    devices: tuple[int, int]
    context_bytes: int
    execution_bytes: int

    def __post_init__(self):
        if len(self.devices) != 2 or len(set(self.devices)) != 2 or any(type(d) is not int or d < 0 for d in self.devices):
            raise ValueError('Exactly two distinct logical devices are required')
        if any(type(n) is not int or n <= 0 for n in (self.host_bytes, self.context_bytes, self.execution_bytes)):
            raise ValueError('Explicit positive per-tier budgets are required')


class ImageRankTransport(Protocol):
    """Absolute deadlines include all peers; stop must kill/reap owned descendants.

    exchange sends a command to both ranks and returns exactly two bounded JSON
    acknowledgements. No tensors/media payloads belong in these control replies.
    start must propagate session/device binding and parent-death cleanup to children.
    """
    def start(self, session: str, devices: tuple[int, int], deadline: float, cancel): ...
    def exchange(self, command: dict, deadline: float, cancel) -> list[dict]: ...
    def stop(self, deadline: float) -> bool: ...


class ImageRankResidency:
    def __init__(self, resources, transport: ImageRankTransport, budget: ImageRankBudget, *,
                 enabled=False, operation_timeout=900, cleanup_timeout=15, clock=time.monotonic):
        if any(not math.isfinite(value) or value <= 0 for value in (operation_timeout, cleanup_timeout)):
            raise ValueError('Finite positive operation and cleanup bounds are required')
        self.resources, self.transport, self.budget = resources, transport, budget
        self.enabled, self.operation_timeout, self.cleanup_timeout = enabled, operation_timeout, cleanup_timeout
        self.clock = clock
        self.gate = threading.RLock()
        self.owner = 'image:ranks:' + uuid4().hex
        self.host = self.context = self.execution = None
        self.state = 'closed'
        self.session = None
        self.sequence = 0
        self.last_job = None
        self.started = False
        self.cleanup_deadline = None

    @contextmanager
    def _ownership(self):
        if not self.gate.acquire(blocking=False):
            raise ResourceBusy('Rank session is active; the global FIFO owns scheduling')
        try:
            if self.state == 'quarantined':
                raise ResourceRecoveryRequired('Both-rank cleanup is unconfirmed; ownership remains reserved')
            yield
        finally:
            self.gate.release()

    def _cancel(self, cancel):
        if cancel is not None and cancel.is_set():
            raise ResourceCancelled('Two-rank operation cancelled')

    def _exchange(self, operation, deadline, cancel=None, job=None, payload=None):
        self._cancel(cancel)
        if self.clock() >= deadline:
            raise TimeoutError('Two-rank operation deadline exhausted')
        self.sequence += 1
        command = dict(version=1, session=self.session, sequence=self.sequence, operation=operation, job=job)
        if payload is not None:
            command["payload"] = payload
        replies = self.transport.exchange(command, deadline, cancel)
        self._cancel(cancel)
        if self.clock() >= deadline:
            raise TimeoutError('Two-rank reply arrived after the deadline')
        if len(replies) != 2 or any(type(r.get('rank')) is not int for r in replies) or {r.get('rank') for r in replies} != {0, 1}:
            raise ValueError('Both unique rank acknowledgements are required')
        for reply in replies:
            if any(reply.get(key) != value for key, value in command.items()):
                raise ValueError('Stale or mismatched rank acknowledgement')
            if reply.get('status') != 'ok' or reply.get('device') != self.budget.devices[reply['rank']]:
                raise ValueError('Rank failed or device ownership differs')
            limit = 0 if operation == 'park' else self.budget.execution_bytes
            resident = reply.get('resident_bytes')
            if type(resident) is not int or not 0 <= resident <= limit:
                raise ValueError('Rank resident bytes exceed admitted execution ownership')
        return replies

    def _reserve_execution(self, cancel):
        self.execution = self.resources.reserve(self.owner + ':execution', 'image',
            device_bytes={device: self.budget.execution_bytes for device in self.budget.devices},
            evict=self.park, cancel_event=cancel)

    def _prepare(self, deadline, cancel):
        if not self.enabled:
            raise ResourceBusy('Two-rank lifecycle is disabled')
        self._cancel(cancel)
        if self.state == 'ready':
            return
        if self.state == 'parked':
            with self.host.lease(cancel), self.context.lease(cancel):
                self._reserve_execution(cancel)
                with self.execution.lease(cancel):
                    self._exchange('restore', deadline, cancel)
            self.state = 'ready'
            return
        if self.state != 'closed':
            raise ResourceBusy('Rank session cannot start from ' + self.state)
        self.state = 'starting'
        self.session = uuid4().hex
        self.sequence = 0
        self.host = self.resources.reserve(self.owner + ':host', 'image', host_bytes=self.budget.host_bytes,
            evict=self.close, cancel_event=cancel)
        with self.host.lease(cancel):
            self.context = self.resources.reserve(self.owner + ':context', 'image',
                device_bytes={device: self.budget.context_bytes for device in self.budget.devices},
                evict=self.close, offload_on_handoff=False, cancel_event=cancel)
            with self.context.lease(cancel):
                self._reserve_execution(cancel)
                with self.execution.lease(cancel):
                    # Mark before start: partial child creation also requires stop/reap.
                    self.started = True
                    self.transport.start(self.session, self.budget.devices, deadline, cancel)
                    self._exchange('ready', deadline, cancel)
        self.state = 'ready'

    def execute(self, job_id, cancel=None, payload=None):
        """The same absolute deadline covers load/restore/execute for one FIFO job."""
        if not re.fullmatch(r'[a-f0-9]{32}', job_id):
            raise ValueError('Use the existing FIFO job identity')
        with self._ownership():
            if self.state in ('running', 'starting'):
                raise ResourceBusy('Reentrant rank execution is forbidden')
            if job_id == self.last_job:
                raise ValueError('Never replay the previous submitted job automatically')
            self.cleanup_deadline = None
            deadline = self.clock() + self.operation_timeout
            try:
                preparing = self.clock()
                self._prepare(deadline, cancel)
                self.prepare_seconds = self.clock()-preparing
                with ExitStack() as leases:
                    for reservation in (self.host, self.context, self.execution):
                        leases.enter_context(reservation.lease(cancel))
                    self.state = 'running'
                    self.last_job = job_id
                    result = self._exchange('execute', deadline, cancel, job_id, payload)
                self.state = 'ready'
                return result
            except BaseException:
                self._stop()
                raise

    def park(self, cancel=None):
        with self._ownership():
            if self.state in ('closed', 'parked'):
                return
            if self.state != 'ready':
                raise ResourceBusy('Cannot park an active rank transaction')
            self.cleanup_deadline = None
            try:
                self._exchange('park', self.clock() + self.operation_timeout, cancel)
                # Both ranks synchronized/freed model allocations. CUDA contexts
                # and reusable CPU weights retain their separate reservations.
                self.execution.release()
                self.execution = None
                self.state = 'parked'
            except ResourceCancelled:
                self._stop()
                raise
            except Exception:
                # RAM reuse is optional. A confirmed cold unload also completes
                # the handoff; uncertain cleanup still raises and retains accounting.
                self._stop()
                logging.getLogger(__name__).warning('Image parking failed; confirmed cold unload completed')

    def _stop(self):
        if self.cleanup_deadline is None:
            self.cleanup_deadline = self.clock() + self.cleanup_timeout
        confirmed = not self.started
        if self.started and self.clock() < self.cleanup_deadline:
            try:
                confirmed = self.transport.stop(self.cleanup_deadline) is True
            except BaseException:
                confirmed = False
        if not confirmed:
            self.state = 'quarantined'
            raise ResourceRecoveryRequired('Rank cleanup unconfirmed; all accounting retained')
        for name in ('execution', 'context', 'host'):
            reservation = getattr(self, name)
            if reservation is not None:
                reservation.release()
                setattr(self, name, None)
        self.started = False
        self.state = 'closed'

    def close(self, *, recover=False):
        # Layered failure cleanup shares one deadline. Only a separately invoked
        # recovery may renew it; eviction and request finalizers must not do so.
        if not self.gate.acquire(blocking=False):
            raise ResourceBusy('Rank session is active')
        try:
            if self.state in ('running', 'starting'):
                raise ResourceBusy('Cannot close an active rank transaction')
            if recover:
                self.cleanup_deadline = None
            self._stop()
        finally:
            self.gate.release()
