"""Conversation-scoped goals and explicit user controls."""

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Request

from ....config import ServiceError
from ....pydantic_models.resources import AgentKey, ResourceKind
from ....pydantic_models.tasks import AssistantTask, TaskControl

router = APIRouter()


@router.get("/api/agents/{key}/conversations/{conversation_id}/tasks")
async def conversation_tasks(
    key: AgentKey, conversation_id: UUID, request: Request
) -> list[AssistantTask]:
    services = request.app.state.services
    await services.resource_of_kinds(key, conversation_id, (ResourceKind.CONVERSATION,))
    values = await services.json_request(
        "Tsuchi", "GET", f"/api/agents/{key}/resources/{conversation_id}/tasks"
    )
    if not isinstance(values, list):
        raise ServiceError("Couldn’t read saved goals. Refresh to try again.", 502)
    tasks = [AssistantTask.model_validate(value) for value in values]
    if any(
        task.agent_key != key or task.conversation_id != conversation_id
        for task in tasks
    ):
        raise ServiceError("Couldn’t read saved goals. Refresh to try again.", 502)
    return tasks


@router.post("/api/agents/{key}/tasks/{task_id}/{action}")
async def control_task(
    key: AgentKey,
    task_id: UUID,
    action: Literal["pause", "resume", "cancel"],
    value: TaskControl,
    request: Request,
) -> AssistantTask:
    task = AssistantTask.model_validate(
        await request.app.state.services.json_request(
            "Tsuchi",
            "POST",
            f"/api/agents/{key}/tasks/{task_id}/{action}",
            json=value.model_dump(mode="json"),
        )
    )
    if task.agent_key != key or task.id != task_id:
        raise ServiceError("Couldn’t confirm the goal update. Refresh to check.", 502)
    return task
