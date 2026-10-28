from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, field_validator

from ...pydantic_models.resources import AgentKey, Title

router = APIRouter()


class AgentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: AgentKey
    name: Title

    @field_validator("name")
    @classmethod
    def nonblank_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Name cannot be blank")
        return value.strip()


class AgentRename(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Title
    expected_name: Title

    @field_validator("name")
    @classmethod
    def nonblank_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Name cannot be blank")
        return value.strip()


@router.get("/api/agents")
async def list_agents(request: Request):
    return await request.app.state.services.json_request("Tsuchi", "GET", "/api/agents")


@router.post("/api/agents", status_code=201)
async def create_agent(value: AgentInput, request: Request):
    return await request.app.state.services.json_request(
        "Tsuchi", "POST", "/api/agents", json=value.model_dump(mode="json")
    )


@router.get("/api/agents/{key}")
async def get_agent(key: AgentKey, request: Request):
    return await request.app.state.services.json_request(
        "Tsuchi", "GET", f"/api/agents/{key}"
    )


@router.put("/api/agents/{key}")
async def rename_agent(key: AgentKey, value: AgentRename, request: Request):
    return await request.app.state.services.json_request(
        "Tsuchi", "PUT", f"/api/agents/{key}", json=value.model_dump(mode="json")
    )
