"""Cancellation that retains ownership until asynchronous or threaded cleanup ends."""

import asyncio


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
