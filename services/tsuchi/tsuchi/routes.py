import json
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, Form, Query, UploadFile
from pydantic import Field, ValidationError, field_validator
from starlette.responses import FileResponse

from .config import StorageError
from .models import (
    Activity,
    Agent,
    AgentInput,
    AgentKey,
    Asset,
    AssetUpdate,
    MessageAppend,
    Metadata,
    Model,
    PetJob,
    PetManifest,
    PetState,
    Resource,
    ResourceInput,
    ResourceKind,
    ResourceUpdate,
    ResourceVersion,
    Stats,
    VersionInput,
    resource_input_adapter,
)
from .repository import Repository
from .services import FileStorage, inline_media, safe_filename


class PetCreate(Model):
    id: UUID
    prompt: str = Field(min_length=1, max_length=10000)

    @field_validator("prompt")
    @classmethod
    def nonblank_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Describe the pet to create")
        return value.strip()


class PetImport(Model):
    id: UUID
    prompt: str = Field(min_length=1, max_length=10000)
    manifest: PetManifest

    @field_validator("prompt")
    @classmethod
    def nonblank_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Name the imported pet")
        return value.strip()


class PetRevision(Model):
    id: UUID
    expected_version: int = Field(ge=1)
    frame_names: list[str] = Field(min_length=1, max_length=73)
    reason: str = Field(min_length=1, max_length=10000)

    @field_validator("reason")
    @classmethod
    def nonblank_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Describe why the frames need repair")
        return value.strip()


class PetUpdate(PetState):
    expected_version: int = Field(ge=1)
    worker_id: UUID | None = None


class PetSelection(Model):
    pet_id: UUID | None


class ExportResponse(FileResponse):
    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.path.unlink(missing_ok=True)


def create_router(repository: Repository, files: FileStorage) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/health")
    def health() -> dict[str, str]:
        repository.health()
        return {"status": "ok", "version": "0.1.0"}

    @router.get("/agents", response_model=list[Agent])
    def list_agents():
        return repository.list_agents()

    @router.post("/agents", response_model=Agent, status_code=201)
    def create_agent(value: AgentInput):
        return repository.create_agent(value)

    @router.get("/pet-jobs", response_model=list[PetJob])
    def active_pet_jobs():
        return repository.active_pet_jobs()

    @router.get("/agents/{key}/pets", response_model=list[PetJob])
    def list_pets(key: AgentKey):
        return repository.list_pets(key)

    @router.post("/agents/{key}/pets", response_model=PetJob, status_code=201)
    def create_pet(key: AgentKey, value: PetCreate):
        return repository.create_pet(key, value.id, value.prompt)

    @router.post("/agents/{key}/pets/imports", response_model=PetJob, status_code=201)
    def import_pet(key: AgentKey, value: PetImport):
        return repository.import_pet(key, value.id, value.prompt, value.manifest)

    @router.post(
        "/agents/{key}/pets/{pet_id}/revisions", response_model=PetJob, status_code=201
    )
    def revise_pet(key: AgentKey, pet_id: UUID, value: PetRevision):
        return repository.revise_pet(
            key,
            pet_id,
            value.id,
            value.expected_version,
            value.frame_names,
            value.reason,
        )

    @router.get("/agents/{key}/pets/{pet_id}", response_model=PetJob)
    def get_pet(key: AgentKey, pet_id: UUID):
        return repository.get_pet(key, pet_id)

    @router.put("/agents/{key}/pets/{pet_id}", response_model=PetJob)
    def update_pet(key: AgentKey, pet_id: UUID, value: PetUpdate):
        state = PetState.model_validate(
            value.model_dump(exclude={"expected_version", "worker_id"})
        )
        return repository.update_pet(
            key, pet_id, state, value.expected_version, value.worker_id
        )

    @router.get("/agents/{key}/pet", response_model=PetJob | None)
    def selected_pet(key: AgentKey):
        return repository.selected_pet(key)

    @router.put("/agents/{key}/pet", response_model=PetJob | None)
    def select_pet(key: AgentKey, value: PetSelection):
        return repository.select_pet(key, value.pet_id)

    @router.get("/agents/{key}/stats", response_model=Stats)
    def stats(key: AgentKey):
        return repository.stats(key)

    @router.get("/agents/{key}/activity", response_model=list[Activity])
    def activity(key: AgentKey, limit: Annotated[int, Query(ge=1, le=200)] = 50):
        return repository.activity(key, limit)

    @router.get("/agents/{key}/resources", response_model=list[Resource])
    def list_resources(
        key: AgentKey,
        kind: ResourceKind | None = None,
        q: str = "",
        trash: bool = False,
        limit: Annotated[int, Query(ge=1, le=200)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ):
        return repository.list_resources(key, kind, q, trash, limit, offset)

    @router.post("/agents/{key}/resources", response_model=Resource, status_code=201)
    def create_resource(key: AgentKey, value: ResourceInput):
        return repository.create_resource(key, value)

    @router.get("/agents/{key}/resources/{resource_id}", response_model=Resource)
    def get_resource(key: AgentKey, resource_id: UUID):
        return repository.get_resource(key, resource_id)

    @router.put("/agents/{key}/resources/{resource_id}", response_model=Resource)
    def update_resource(key: AgentKey, resource_id: UUID, value: ResourceUpdate):
        content = resource_input_adapter.validate_python(
            value.model_dump(exclude={"expected_version"})
        )
        return repository.update_resource(
            key, resource_id, content, value.expected_version
        )

    @router.delete("/agents/{key}/resources/{resource_id}", response_model=Resource)
    def trash_resource(
        key: AgentKey, resource_id: UUID, expected_version: Annotated[int, Query(ge=1)]
    ):
        return repository.trash_resource(key, resource_id, expected_version)

    @router.post(
        "/agents/{key}/resources/{resource_id}/restore", response_model=Resource
    )
    def restore_resource(key: AgentKey, resource_id: UUID, value: VersionInput):
        return repository.restore_resource(key, resource_id, value.expected_version)

    @router.get(
        "/agents/{key}/resources/{resource_id}/versions",
        response_model=list[ResourceVersion],
    )
    def versions(key: AgentKey, resource_id: UUID):
        return repository.versions(key, resource_id)

    @router.post(
        "/agents/{key}/resources/{resource_id}/versions/{version}/restore",
        response_model=Resource,
    )
    def restore_version(
        key: AgentKey, resource_id: UUID, version: int, value: VersionInput
    ):
        return repository.restore_version(
            key, resource_id, version, value.expected_version
        )

    @router.post(
        "/agents/{key}/resources/{resource_id}/messages", response_model=Resource
    )
    def append_message(key: AgentKey, resource_id: UUID, value: MessageAppend):
        return repository.append_message(
            key, resource_id, value.message, value.expected_version
        )

    @router.get("/agents/{key}/assets", response_model=list[Asset])
    def list_assets(
        key: AgentKey, images: bool = False, trash: bool = False, q: str = ""
    ):
        return repository.list_assets(key, images, trash, q)

    @router.post("/agents/{key}/assets", response_model=Asset, status_code=201)
    def upload_asset(
        key: AgentKey,
        file: Annotated[UploadFile, File()],
        title: Annotated[str | None, Form()] = None,
        tags: Annotated[str, Form()] = "[]",
        sources: Annotated[str, Form()] = "[]",
        asset_id: Annotated[UUID | None, Form()] = None,
    ):
        try:
            metadata = Metadata(
                title=title or safe_filename(file.filename),
                tags=json.loads(tags),
                sources=json.loads(sources),
            )
        except (json.JSONDecodeError, ValidationError) as error:
            raise StorageError(
                "Upload title, tags or sources are invalid", 422
            ) from error
        return files.upload(key, file.file, file.filename, metadata, asset_id)

    @router.get("/agents/{key}/assets/{asset_id}", response_model=Asset)
    def get_asset(key: AgentKey, asset_id: UUID):
        return repository.get_asset(key, asset_id)

    @router.get("/agents/{key}/assets/{asset_id}/content")
    def asset_content(key: AgentKey, asset_id: UUID, download: bool = False):
        asset = repository.get_asset(key, asset_id)
        path = files.blob_path(asset.id)
        if not path.is_file():
            raise StorageError("Stored file is missing", 404)
        return FileResponse(
            path,
            media_type=asset.media_type,
            filename=asset.filename,
            content_disposition_type="inline"
            if inline_media(asset) and not download
            else "attachment",
            headers={
                "Cache-Control": "private, no-store",
                "Content-Security-Policy": "default-src 'none'; sandbox",
            },
        )

    @router.put("/agents/{key}/assets/{asset_id}", response_model=Asset)
    def update_asset(key: AgentKey, asset_id: UUID, value: AssetUpdate):
        metadata = Metadata.model_validate(
            value.model_dump(exclude={"expected_version"})
        )
        return repository.update_asset(key, asset_id, metadata, value.expected_version)

    @router.delete("/agents/{key}/assets/{asset_id}", response_model=Asset)
    def trash_asset(
        key: AgentKey, asset_id: UUID, expected_version: Annotated[int, Query(ge=1)]
    ):
        return repository.trash_asset(key, asset_id, expected_version)

    @router.post("/agents/{key}/assets/{asset_id}/restore", response_model=Asset)
    def restore_asset(key: AgentKey, asset_id: UUID, value: VersionInput):
        return repository.restore_asset(key, asset_id, value.expected_version)

    @router.get("/agents/{key}/export")
    def export(key: AgentKey):
        path = files.export(key)
        return ExportResponse(
            path, media_type="application/zip", filename=f"tsuchi-{key}.zip"
        )

    return router
