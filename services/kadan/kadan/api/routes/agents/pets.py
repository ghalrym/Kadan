import asyncio
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, Form, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from ...config import ServiceError
from ...pet_import import MAX_FILE_BYTES, read_package
from ...pydantic_models.pets import PetJob, PetReferenceMode
from ...pydantic_models.resources import AgentKey

router = APIRouter()


class CreatePet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: UUID
    prompt: str = Field(min_length=1, max_length=4000, pattern=r"\S")
    reference_mode: PetReferenceMode = "single_image"


class SelectPet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pet_id: UUID | None


class RevisePet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: UUID
    expected_version: int = Field(ge=1)
    frame_names: list[str] = Field(min_length=1, max_length=73)
    reason: str = Field(min_length=1, max_length=2000, pattern=r"\S")


@router.post("/api/agents/{key}/pets/{pet_id}/revisions", status_code=202)
async def revise_pet(
    key: AgentKey, pet_id: UUID, value: RevisePet, request: Request
) -> PetJob:
    return await request.app.state.services.pets.revise(
        key, pet_id, value.id, value.expected_version, value.frame_names, value.reason
    )


@router.get("/api/agents/{key}/pets")
async def list_pets(key: AgentKey, request: Request) -> list[PetJob]:
    return await request.app.state.services.pets.list(key)


@router.post("/api/agents/{key}/pets", status_code=202)
async def create_pet(key: AgentKey, value: CreatePet, request: Request) -> PetJob:
    return await request.app.state.services.pets.create(
        key, value.id, value.prompt, value.reference_mode
    )


@router.post("/api/agents/{key}/pets/imports", status_code=201)
async def import_pet(
    key: AgentKey,
    request: Request,
    id: Annotated[UUID, Form()],
    files: Annotated[list[UploadFile], File()],
    name: Annotated[str, Form(max_length=200)] = "",
) -> PetJob:
    """Import a Codex pet as a ready candidate. Selection stays explicit."""
    if len(files) > 3:
        raise ServiceError(
            "Upload a zip, or a spritesheet with its pet.json.",
            422,
            "invalid_pet_package",
        )
    uploads = []
    for upload in files:
        data = await upload.read(MAX_FILE_BYTES + 1)
        uploads.append((upload.filename or "upload", data))
    try:
        pet = await asyncio.to_thread(read_package, uploads)
    except ValueError as error:
        raise ServiceError(str(error), 422, "invalid_pet_package") from error
    if name.strip():
        pet.name = name.strip()
    return await request.app.state.services.pets.import_atlas(
        key, id, pet.prompt, pet.atlas, pet.sprite_version
    )


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
