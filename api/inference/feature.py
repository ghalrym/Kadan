"""Small native feature contract; queueing belongs to MemoryManager.

Adapters keep their existing ResourceManager reservations. These wrappers hold
one service/adapter handle and never maintain a second residency ledger.
"""
import asyncio
from contextlib import suppress
import threading
from typing import Protocol, runtime_checkable

from api.services.runtime import RuntimeFailure, finish_cleanup


@runtime_checkable
class InferenceFeature(Protocol):
    name: str
    workload: str
    @property
    def adapter(self): ...
    async def __call__(self, request, *, model=None, operation='generate', job_id=None): ...
    async def load(self, model=None): ...
    async def offload_to_ram(self): ...
    async def unload(self): ...


async def native_call(function, *args):
    """Wait for actual native cleanup before handing ownership to another job."""
    cancel = threading.Event()
    worker = asyncio.create_task(asyncio.to_thread(function, *args, cancel))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancel.set()
        with suppress(Exception, asyncio.CancelledError):
            await finish_cleanup(worker)
        raise


class UnsupportedFeature:
    adapter = None
    def __init__(self, message):
        self.message = message
    async def load(self, model=None):
        raise RuntimeFailure(self.message)
    async def __call__(self, request, *, model=None, operation='generate', job_id=None):
        raise RuntimeFailure(self.message)
    async def offload_to_ram(self):
        return None  # No model exists to move or discard.
    async def unload(self):
        return None
