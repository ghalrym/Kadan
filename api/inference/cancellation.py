"""Cancellation that retains ownership until asynchronous or threaded cleanup ends."""
import asyncio
from contextlib import suppress
import threading


async def await_cleanup(task):
    """Keep ownership until cleanup ends, then propagate any caller cancellation."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


async def run_cancellable_thread(function, *args):
    """Wait for actual thread cleanup before handing ownership to another job."""
    cancel = threading.Event()
    worker = asyncio.create_task(asyncio.to_thread(function, *args, cancel))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancel.set()
        with suppress(Exception, asyncio.CancelledError):
            await await_cleanup(worker)
        raise
