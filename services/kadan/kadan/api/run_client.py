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
            attachment_ids=value.attachment_ids,
            resource_ids=[value.persona_id] if value.persona_id else [],
        )
        body = {
            "message": message.model_dump(mode="json"),
            "expected_version": value.expected_version,
            "references": [item.model_dump(mode="json") for item in value.references],
            "persona_id": str(value.persona_id) if value.persona_id else None,
        }
        saved = await self.services.json_request(
            "Tsuchi",
            "POST",
            f"/api/agents/{key}/resources/{conversation_id}/runs",
            json=body,
        )
        run = AgentRun.model_validate(saved)
        if (run.agent_key, run.conversation_id, run.id) != (
            key,
            conversation_id,
            value.message_id,
        ):
            raise ServiceError(
                "Storage did not confirm the requested response work. Inspect Messages before retrying.",
                502,
            )
        return run

    async def cancel(self, key: str, run_id: UUID) -> AgentRun:
        return AgentRun.model_validate(
            await self.services.json_request(
                "Tsuchi", "POST", self.path(key, run_id) + "/cancel", json={}
            )
        )
