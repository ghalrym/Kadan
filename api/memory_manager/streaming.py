"""Bounded thread-to-HTTP channel; Redis still owns completion and cancellation."""
import asyncio
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import suppress
import threading
import time

from api.services.runtime import finish_cleanup


class QueuedStream:
    def __init__(self, queue):
        self.queue, self.job_id = queue, None
        self.loop = asyncio.get_running_loop()
        self.events = asyncio.Queue(maxsize=16)
        self.closed = threading.Event()
        self.waiter = None
        self.published = False

    def start(self):
        self.waiter = asyncio.create_task(self.queue.wait(self.job_id))

    def emit(self, event):
        """Called by the existing native thread, with bounded backpressure and disconnect escape."""
        if self.closed.is_set():
            raise InterruptedError('Stream disconnected')
        self.published = True
        pending = asyncio.run_coroutine_threadsafe(self.events.put(event), self.loop)
        deadline = time.monotonic() + 10
        try:
            while True:
                try:
                    return pending.result(timeout=.05)
                except FutureTimeout:
                    if self.closed.is_set():
                        raise InterruptedError('Stream disconnected')
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Chat stream consumer is too slow.')
        finally:
            if not pending.done():
                pending.cancel()

    async def __aiter__(self):
        while True:
            # Completion is authoritative: no successful terminal event escapes
            # before native cleanup and Redis result validation have completed.
            if self.waiter.done() and self.events.empty():
                await self.waiter
                return
            try:
                event = await asyncio.wait_for(self.events.get(), .25)
            except TimeoutError:
                continue
            yield event

    async def aclose(self):
        self.closed.set()
        if self.waiter is not None:
            if not self.waiter.done():
                self.waiter.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await finish_cleanup(self.waiter)
        if self.job_id is not None:
            self.queue.streams.pop(self.job_id, None)
