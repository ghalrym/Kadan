from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...pydantic_models.resources import AgentKey, Resource, Title

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


class OwnedPersonaInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = Field(default=None, min_length=1, max_length=200)
    markdown: str = Field(min_length=1, max_length=16000)
    expected_version: int | None = Field(default=None, ge=1)

    @field_validator("title")
    @classmethod
    def nonblank_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("Title cannot be blank")
        return value.strip()

    @field_validator("markdown")
    @classmethod
    def nonblank_markdown(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Persona instructions cannot be blank")
        return value


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


@router.get("/api/agents/{key}/persona", response_model=Resource | None)
async def get_persona(key: AgentKey, request: Request) -> Resource | None:
    return await request.app.state.services.agent_persona(key)


@router.put("/api/agents/{key}/persona", response_model=Resource)
async def save_persona(
    key: AgentKey, value: OwnedPersonaInput, request: Request
) -> Resource:
    return await request.app.state.services.save_agent_persona(
        key,
        markdown=value.markdown,
        expected_version=value.expected_version,
        title=value.title,
    )
