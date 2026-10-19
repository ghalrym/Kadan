from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict

from ..pydantic_models.resources import AgentKey
from ..pydantic_models.runners import Runner

router = APIRouter()


class AgentAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runner_id: AgentKey | None
    expected_runner_id: AgentKey | None


@router.get("/api/runners")
async def list_runners(request: Request) -> list[Runner]:
    values = await request.app.state.services.json_request(
        "Tsuchi", "GET", "/api/runners"
    )
    return [Runner.model_validate(value) for value in values]


@router.post("/api/agents/{key}/runner", status_code=204)
async def reassign_agent(
    key: AgentKey, value: AgentAssignment, request: Request
) -> None:
    await request.app.state.services.json_request(
        "Tsuchi",
        "POST",
        f"/api/agents/{key}/runner",
        json=value.model_dump(mode="json"),
    )
