"""API management of durable work; execution belongs to the runner service."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from .config import ServiceError
from .pydantic_models.agent_runs import AgentRun
from .pydantic_models.resources import Message

if TYPE_CHECKING:
    from .routes.agents.conversations.runs import RunInput
    from .services import LocalServices


class RunClient:
    def __init__(self, services: LocalServices):
        self.services = services

    @staticmethod
    def path(key: str, run_id: UUID) -> str:
        return f"/api/agents/{key}/runs/{run_id}"

    async def get(self, key: str, run_id: UUID) -> AgentRun:
        run = AgentRun.model_validate(
            await self.services.json_request("Tsuchi", "GET", self.path(key, run_id))
        )
        if run.agent_key != key or run.id != run_id:
            raise ServiceError("Tsuchi returned the wrong agent run.", 502)
        return run

    async def enqueue(
        self, key: str, conversation_id: UUID, value: RunInput
    ) -> AgentRun:
        message = Message(
            id=value.message_id,
            role="user",
            content=value.content,
            resource_ids=[value.persona_id] if value.persona_id else [],
        )
        return AgentRun.model_validate(
            await self.services.json_request(
                "Tsuchi",
                "POST",
                f"/api/agents/{key}/resources/{conversation_id}/runs",
                json={
                    "message": message.model_dump(mode="json"),
                    "expected_version": value.expected_version,
                    "references": [
                        item.model_dump(mode="json") for item in value.references
                    ],
                    "workflow_id": str(value.workflow_id)
                    if value.workflow_id
                    else None,
                    "workflow_version": value.workflow_version,
                    "persona_id": str(value.persona_id) if value.persona_id else None,
                },
            )
        )

    async def cancel(self, key: str, run_id: UUID) -> AgentRun:
        return AgentRun.model_validate(
            await self.services.json_request(
                "Tsuchi", "POST", self.path(key, run_id) + "/cancel", json={}
            )
        )
