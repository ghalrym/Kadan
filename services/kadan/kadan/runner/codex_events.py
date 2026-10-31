"""Wake the agent in the conversation that started a Codex run once it finishes."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING
from uuid import uuid5

from ..api.coding_client import CodingClient
from ..api.config import ServiceError
from ..api.pydantic_models.coding import CodexFinished

if TYPE_CHECKING:
    from ..api.services import LocalServices

logger = logging.getLogger(__name__)
POLL_SECONDS = 5


def event_content(run: CodexFinished) -> str:
    return json.dumps(
        {
            "thread_id": str(run.thread_id),
            "status": run.status,
            "cwd": run.cwd,
            "message": run.message,
        },
        ensure_ascii=False,
    )


class CodexEvents:
    def __init__(self, services: LocalServices):
        self.services = services
        self.runner: asyncio.Task | None = None

    def start(self) -> None:
        self.runner = asyncio.create_task(self.run(), name="codex-events")

    async def close(self) -> None:
        if self.runner:
            self.runner.cancel()
            await asyncio.gather(self.runner, return_exceptions=True)

    async def run(self) -> None:
        while True:
            if self.services.sandbox.configured:
                try:
                    await self.deliver()
                except ServiceError as error:
                    logger.warning("Codex results could not be delivered: %s", error)
            await asyncio.sleep(POLL_SECONDS)

    async def deliver(self) -> None:
        client = CodingClient(self.services.sandbox)
        failure = None
        for finished in await client.finished():
            try:
                await self.deliver_run(client, finished)
            except ServiceError as error:
                failure = error
                logger.warning(
                    "Codex result for %s could not be delivered: %s",
                    finished.request_id,
                    error,
                )
        if failure is not None:
            raise failure

    async def deliver_run(self, client: CodingClient, finished: CodexFinished) -> None:
        origin = finished.origin
        if origin is not None:
            # The event run ID derives from the Codex run, so a crash before
            # acknowledging repeats this request without queuing a second run.
            try:
                await self.services.json_request(
                    "Tsuchi",
                    "POST",
                    f"/api/agents/{origin.agent_key}/resources/"
                    f"{origin.conversation_id}/event-runs",
                    json={
                        "id": str(uuid5(finished.request_id, "codex-event")),
                        "origin_run_id": str(origin.run_id),
                        "event": {
                            "kind": "codex",
                            "content": event_content(finished),
                        },
                    },
                )
            except ServiceError as error:
                if error.status != 404 and not (
                    error.status == 409
                    and str(error) == "Restore this item from trash before editing it"
                ):
                    raise
                # The conversation is gone or no longer accepts runs.
                logger.warning(
                    "Codex result for thread %s was dropped: %s",
                    finished.thread_id,
                    error,
                )
        await client.acknowledge(finished.request_id)
