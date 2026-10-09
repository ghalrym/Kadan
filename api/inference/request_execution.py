"""Shared inference request execution contract; queueing belongs to MemoryManager.

Adapters keep their existing ResourceManager reservations. These wrappers hold
one model evaluator handle and never maintain a second residency ledger.
"""
from typing import Protocol, runtime_checkable

from api.inference.errors import InferenceFailure


@runtime_checkable
class RequestExecutor(Protocol):
    name: str
    workload: str
    @property
    def adapter(self): ...
    async def __call__(self, request, *, model=None, operation='generate', job_id=None): ...
    async def load(self, model=None): ...
    async def offload_to_ram(self): ...
    async def unload(self): ...


class UnavailableRequestExecutor:
    adapter = None
    def __init__(self, message):
        self.message = message
    async def load(self, model=None):
        raise InferenceFailure(self.message)
    async def __call__(self, request, *, model=None, operation='generate', job_id=None):
        raise InferenceFailure(self.message)
    async def offload_to_ram(self):
        return None  # No model exists to move or discard.
    async def unload(self):
        return None
