"""Bounded process-local stage visibility for the single shared queue owner."""
from contextlib import contextmanager
from contextvars import ContextVar
import threading
import time

_current = ContextVar('inference_progress', default=None)
_lock = threading.Lock()
_active = None


@contextmanager
def track(job_id, workload):
    global _active
    state = dict(job_id=job_id, workload=workload, stage='preparing', value=0, observed_unix_ns=time.time_ns())
    token = _current.set(state)
    with _lock:
        _active = state
    try:
        yield
    finally:
        with _lock:
            if _active is state:
                _active = None
        _current.reset(token)


def report(stage, value=0):
    state = _current.get()
    if state is not None:
        with _lock:
            state.update(stage=stage, value=int(value), observed_unix_ns=time.time_ns())


def snapshot():
    with _lock:
        return dict(_active) if _active is not None else None
