from uuid import UUID

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from ...pydantic_models.computer import ComputerRequest, ComputerTask, ComputerThread
from ...pydantic_models.resources import AgentKey

router = APIRouter(prefix="/api/agents/{key}/computer")


class ComputerRequestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: UUID
    content: str = Field(min_length=1, max_length=10000, pattern=r"\S")


class ComputerWork(BaseModel):
    """What the panel shows; the computer agent's transcript stays private."""

    task: ComputerTask | None
    queued: int
    pending_steering: int
    # The most recently finished task, while nothing is active.
    last: ComputerTask | None = None


@router.get("")
async def computer_work(key: AgentKey, request: Request) -> ComputerWork:
    services = request.app.state.services
    tasks = [
        ComputerTask.model_validate(value)
        for value in await services.json_request(
            "Tsuchi", "GET", f"/api/agents/{key}/computer-tasks"
        )
    ]
    thread = ComputerThread.model_validate(
        await services.json_request(
            "Tsuchi", "GET", f"/api/agents/{key}/computer-thread"
        )
    )
    last = None
    if not tasks:
        latest = await services.json_request(
            "Tsuchi", "GET", f"/api/agents/{key}/computer-tasks/latest"
        )
        last = ComputerTask.model_validate(latest) if latest else None
    return ComputerWork(
        task=tasks[0] if tasks else None,
        queued=max(0, len(tasks) - 1),
        pending_steering=len(thread.steering),
        last=last,
    )


@router.post("/requests", status_code=201)
async def computer_request(
    key: AgentKey, value: ComputerRequestInput, request: Request
) -> ComputerRequest:
    """Steer the running task, or start a new one when the computer is idle."""
    return ComputerRequest.model_validate(
        await request.app.state.services.json_request(
            "Tsuchi",
            "POST",
            f"/api/agents/{key}/computer-requests",
            json=value.model_dump(mode="json"),
        )
    )


@router.post("/tasks/{task_id}/cancel")
async def cancel_computer_task(
    key: AgentKey, task_id: UUID, request: Request
) -> ComputerTask:
    return ComputerTask.model_validate(
        await request.app.state.services.json_request(
            "Tsuchi", "POST", f"/api/agents/{key}/computer-tasks/{task_id}/cancel"
        )
    )
