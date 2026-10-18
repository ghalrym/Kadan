"""Await queue results without abandoning native cleanup on client disconnect."""

import asyncio
from contextlib import suppress

import anyio
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from api.services.inference_errors import InferenceFailure

from api.services.inference_cancellation import await_cleanup


async def infer(request, call):
    async def disconnect():
        while True:
            if (await request.receive())["type"] == "http.disconnect":
                return

    generation = asyncio.create_task(call)
    watcher = asyncio.create_task(disconnect())
    try:
        done, _ = await asyncio.wait(
            (generation, watcher), return_when=asyncio.FIRST_COMPLETED
        )
        if generation not in done:
            generation.cancel()
            raise HTTPException(499, "Client disconnected; inference cancelled.")
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


class OwnedStreamResponse(StreamingResponse):
    def __init__(self, content, owner):
        super().__init__(
            content,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
        self.owner = owner

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette uses cancellation scopes on disconnect. Shield cleanup
            # through the queue/native owner, including a disconnect before iteration.
            with anyio.CancelScope(shield=True):
                await self.owner.aclose()
