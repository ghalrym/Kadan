"""API management of durable work; execution belongs to the runner service."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from .config import ServiceError
from .coding_client import CodingClient
from .pydantic_models.agent_runs import AgentRun, RunMessage, ToolCommand
from .pydantic_models.coding import CodingContinuation
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
        coding_task = None
        initial_command = None
        if value.resume_coding_job_id is not None:
            job = await CodingClient(self.services.sandbox).job(
                value.resume_coding_job_id
            )
            if job.status in {"starting", "running", "stopping", "cleanup_required"}:
                raise ServiceError(
                    "Wait for coding cleanup before continuing this session.", 409
                )
            if not job.session_id:
                raise ServiceError(
                    "This coding job has no saved session to continue.", 409
                )
            coding_task = CodingContinuation(
                prompt=value.content, cwd=job.cwd, resume_job_id=job.id
            )
            initial_command = RunMessage(
                role="assistant",
                content=ToolCommand(
                    tool="codex_start", arguments=coding_task.model_dump(mode="json")
                ).model_dump_json(),
            )
        message = Message(
            id=value.message_id,
            role="user",
            content=(
                f"Continue with Codex in `{coding_task.cwd}`:\n\n{value.content}"
                if coding_task
                else value.content
            ),
            resource_ids=[value.persona_id] if value.persona_id else [],
        )
        body = {
            "message": message.model_dump(mode="json"),
            "expected_version": value.expected_version,
            "persona_id": str(value.persona_id) if value.persona_id else None,
        }
        if initial_command is not None:
            body["initial_command"] = initial_command.model_dump(mode="json")
        try:
            saved = await self.services.json_request(
                "Tsuchi",
                "POST",
                f"/api/agents/{key}/resources/{conversation_id}/runs",
                json=body,
            )
        except ServiceError as error:
            if coding_task is not None and error.status == 422:
                raise ServiceError(
                    "Storage could not accept this continuation. Ask the person running Kadan "
                    "to check that Kadan and Tsuchi are updated together. Check Messages before retrying.",
                    503,
                ) from error
            raise
        run = AgentRun.model_validate(saved)
        if (run.agent_key, run.conversation_id, run.id, run.initial_command) != (
            key,
            conversation_id,
            value.message_id,
            initial_command,
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
