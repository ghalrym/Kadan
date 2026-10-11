"""Await queue results without abandoning native cleanup on client disconnect."""
import asyncio
from contextlib import suppress

from fastapi import HTTPException

from api.inference.errors import InferenceFailure

from api.inference.cancellation import await_cleanup


async def infer(request, call):
    async def disconnect():
        while True:
            if (await request.receive())['type'] == 'http.disconnect':
                return

    generation = asyncio.create_task(call)
    watcher = asyncio.create_task(disconnect())
    try:
        done, _ = await asyncio.wait((generation, watcher), return_when=asyncio.FIRST_COMPLETED)
        if generation not in done:
            generation.cancel()
            raise HTTPException(499, 'Client disconnected; inference cancelled.')
        return await generation
    except InferenceFailure as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    finally:
        watcher.cancel()
        if not generation.done():
            generation.cancel()
        for task in (generation, watcher):
            with suppress(asyncio.CancelledError, Exception):
                await await_cleanup(task)
