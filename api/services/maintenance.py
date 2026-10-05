"""Atomic admission and real background-work ownership for a single API worker.

The host helper also holds a read-only maintenance file across container restarts.
An HTTP snapshot is never used to establish quiescence. Tickets outlive background
downloads/load responses and are released only after their actual cleanup ends.
"""
import os
from pathlib import Path
import threading
from contextvars import ContextVar

from starlette.responses import JSONResponse


class MaintenanceBusy(RuntimeError):
    pass


restoring = ContextVar('maintenance_restore', default=False)


class Ticket:
    def __init__(self, gate):
        self.gate = gate
        self.closed = False

    def close(self):
        with self.gate.lock:
            if not self.closed:
                self.gate.active -= 1
                self.closed = True


class Admission:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.blocked = False

    def enter(self):
        with self.lock:
            marker = os.environ.get('KADAN_MAINTENANCE_FILE')
            if not restoring.get() and (self.blocked or (marker and Path(marker).exists())):
                raise MaintenanceBusy('Kadan is preparing an update; retry after it reconnects.')
            self.active += 1
            return Ticket(self)

    def seal(self):
        with self.lock:
            self.blocked = True

    def resume(self):
        with self.lock:
            self.blocked = False

    def count(self):
        with self.lock:
            return self.active


admission = Admission()


class MaintenanceMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (scope['type'] != 'http' or scope['method'] in ('GET', 'HEAD', 'OPTIONS')
                or scope['path'].startswith(('/internal/updates/', '/v1/updates'))):
            return await self.app(scope, receive, send)
        try:
            ticket = admission.enter()
        except MaintenanceBusy as error:
            return await JSONResponse({'detail': str(error)}, status_code=503)(scope, receive, send)
        try:
            await self.app(scope, receive, send)
        finally:
            ticket.close()
