from uuid import UUID

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from ...pydantic_models.pets import PetJob
from ...pydantic_models.resources import AgentKey

router = APIRouter()


class CreatePet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: UUID
    prompt: str = Field(min_length=1, max_length=4000, pattern=r"\S")


class SelectPet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pet_id: UUID | None


@router.get("/api/agents/{key}/pets")
async def list_pets(key: AgentKey, request: Request) -> list[PetJob]:
    return await request.app.state.services.pets.list(key)


@router.post("/api/agents/{key}/pets", status_code=202)
async def create_pet(key: AgentKey, value: CreatePet, request: Request) -> PetJob:
    return await request.app.state.services.pets.create(key, value.id, value.prompt)


@router.get("/api/agents/{key}/pets/{pet_id}")
async def pet_status(key: AgentKey, pet_id: UUID, request: Request) -> PetJob:
    return await request.app.state.services.pets.get(key, pet_id)


@router.post("/api/agents/{key}/pets/{pet_id}/cancel", status_code=202)
async def cancel_pet(key: AgentKey, pet_id: UUID, request: Request) -> PetJob:
    return await request.app.state.services.pets.cancel(key, pet_id)


@router.post("/api/agents/{key}/pets/{pet_id}/resume", status_code=202)
async def resume_pet(key: AgentKey, pet_id: UUID, request: Request) -> PetJob:
    return await request.app.state.services.pets.resume(key, pet_id)


@router.get("/api/agents/{key}/pet")
async def selected_pet(key: AgentKey, request: Request) -> PetJob | None:
    return await request.app.state.services.pets.selected(key)


@router.put("/api/agents/{key}/pet")
async def select_pet(
    key: AgentKey, value: SelectPet, request: Request
) -> PetJob | None:
    # Only the explicit UI action exposes selection. MCP creation tools cannot activate.
    return await request.app.state.services.pets.select(key, value.pet_id)
