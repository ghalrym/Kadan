import json
import threading
from datetime import datetime
from typing import Annotated, Literal
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
    AgentRename,
    AgentRun,
    AgentRunState,
    Asset,
    AssetMetadata,
    AssetUpdate,
    ChatResultInput,
    ConversationCompactionUpdate,
    ConversationMemoryStateUpdate,
    ConversationTaskPage,
    GroupChatDeliveryStatus,
    GroupChatEventSubscription,
    GroupChatEventSubscriptionInput,
    GroupChatEventSubscriptionUpdate,
    GroupChatMessageInput,
    GroupChatMessagesPage,
    GroupChatMessageWriteResult,
    GroupChatOAuthClient,
    GroupChatOAuthClientInput,
    GroupChatOAuthCode,
    GroupChatOAuthCodeConsume,
    GroupChatOAuthCodeInput,
    GroupChatOAuthToken,
    GroupChatOAuthTokenInput,
    GroupChatOAuthTokenRefresh,
    GroupChatParticipant,
    GroupChatParticipantInput,
    GroupChatParticipantTouch,
    GroupChatParticipantUpdate,
    GroupChatRoom,
    GroupChatRoomUpdate,
    GroupChatSubscriptionStatus,
    GroupChatTokenAuth,
    GroupChatWebhookDelivery,
    GroupChatWebhookDeliveryUpdate,
    ImageJob,
    ImageJobState,
    MarkdownResource,
    McpOAuthState,
    McpServer,
    McpServerInput,
    McpServerUpdate,
    Message,
    MessageAppend,
    Model,
    OwnedPersonaUpdate,
    PetJob,
    PetManifest,
    PetReferenceMode,
    PetState,
    RequestReference,
    Resource,
    ResourceInput,
    ResourceKind,
    ResourceUpdate,
    ResourceVersion,
    RunEvent,
    Runner,
    Stats,
    T3ThreadWatch,
    T3ThreadWatchInput,
    T3ThreadWatchUpdate,
    UserProfile,
    UserProfileUpdate,
    VersionInput,
    WorkActivityInput,
    WorkActivityItem,
    WorkBucket,
    WorkHarnessConfig,
    WorkHarnessConfigInput,
    WorkHarnessId,
    WorkPendingRequest,
    WorkPendingRequestResponse,
    WorkProject,
    WorkProjectInput,
    WorkProjectUpdate,
    WorkRun,
    WorkRunInput,
    WorkRunUpdate,
    WorkThread,
    WorkThreadInput,
    WorkThreadOrganize,
    WorkThreadUpdate,
    resource_input_adapter,
)
from .repository import Repository
from .services import FileStorage, inline_media, safe_filename


class PetCreate(Model):
    id: UUID
    prompt: str = Field(min_length=1, max_length=10000)
    reference_mode: PetReferenceMode = "template_strips"

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


class RunCreate(Model):
    expected_version: int = Field(ge=1)
    message: Message
    persona_id: UUID | None = None
    references: list[RequestReference] = Field(default_factory=list, max_length=3)


class EventRunCreate(Model):
    id: UUID
    origin_run_id: UUID | None = None
    event: RunEvent


class RunUpdate(AgentRunState):
    transcript_format: Literal[2]
    expected_version: int = Field(ge=1)
    worker_id: UUID


class RunDelivery(Model):
    worker_id: UUID
    message: Message
    image_job_id: UUID | None = None


class RunDraftUpdate(Model):
    worker_id: UUID
    content: str = Field(max_length=48000)


class ImageJobCreate(Model):
    id: UUID
    prompt: str = Field(min_length=1, max_length=10000)
    request_id: UUID
    worker_id: UUID


class ImageJobUpdate(ImageJobState):
    expected_version: int = Field(ge=1)
    worker_id: UUID


class ExportResponse(FileResponse):
    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.path.unlink(missing_ok=True)


class RunnerRegistration(Model):
    worker_id: UUID
    capacity: int = Field(default=5, ge=1, strict=True)
    expected_worker_id: UUID | None = None


class RunnerAssignment(Model):
    worker_id: UUID


class AgentRunnerAssignment(Model):
    runner_id: AgentKey | None
    expected_runner_id: AgentKey | None


def create_router(repository: Repository, files: FileStorage) -> APIRouter:
    router = APIRouter(prefix="/api")
    draft_lock = threading.Lock()
    run_drafts: dict[tuple[str, UUID], str] = {}

    def clear_draft(key: str, run_id: UUID) -> None:
        with draft_lock:
            run_drafts.pop((key, run_id), None)

    def clear_agent_drafts(key: str) -> None:
        with draft_lock:
            for draft_key in [entry for entry in run_drafts if entry[0] == key]:
                run_drafts.pop(draft_key, None)

    def with_draft(key: str, run: AgentRun) -> AgentRun:
        if run.status != "running":
            clear_draft(key, run.id)
            return run
        with draft_lock:
            content = run_drafts.get((key, run.id))
        if not content:
            return run
        return run.model_copy(update={"draft": content})

    @router.get("/runners", response_model=list[Runner])
    def list_runners():
        return repository.list_runners()

    @router.put("/runners/{runner_id}", response_model=Runner)
    def register_runner(runner_id: AgentKey, value: RunnerRegistration):
        return repository.register_runner(runner_id, **value.model_dump())

    @router.post("/runners/{runner_id}/assign", response_model=Runner)
    def assign_runner(runner_id: AgentKey, value: RunnerAssignment):
        return repository.assign_runner(runner_id, value.worker_id)

    @router.post("/agents/{key}/runner", status_code=204)
    def reassign_agent(key: AgentKey, value: AgentRunnerAssignment):
        repository.reassign_agent(key, **value.model_dump())

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

    @router.get("/agents/{key}/persona", response_model=MarkdownResource | None)
    def get_owned_persona(key: AgentKey):
        return repository.get_owned_persona(key)

    @router.put("/agents/{key}/persona", response_model=MarkdownResource)
    def save_owned_persona(key: AgentKey, value: OwnedPersonaUpdate):
        return repository.save_owned_persona(key, value)

    @router.get("/agents/{key}", response_model=Agent)
    def get_agent(key: AgentKey):
        return repository.get_agent(key)

    @router.put("/agents/{key}", response_model=Agent)
    def rename_agent(key: AgentKey, value: AgentRename):
        return repository.rename_agent(key, value)

    @router.delete("/agents/{key}", status_code=204)
    def delete_agent(key: AgentKey):
        files.delete_agent(key)
        clear_agent_drafts(key)

    @router.get("/agent-runs", response_model=list[AgentRun])
    def active_runs():
        return repository.active_runs()

    @router.get(
        "/agents/{key}/resources/{conversation_id}/runs", response_model=list[AgentRun]
    )
    def conversation_runs(key: AgentKey, conversation_id: UUID):
        return [
            with_draft(key, run)
            for run in repository.conversation_runs(key, conversation_id)
        ]

    @router.post(
        "/agents/{key}/resources/{conversation_id}/runs",
        response_model=AgentRun,
        status_code=201,
    )
    def create_run(key: AgentKey, conversation_id: UUID, value: RunCreate):
        return repository.create_run(
            key,
            conversation_id,
            value.message,
            value.expected_version,
            value.persona_id,
            value.references,
        )

    @router.post(
        "/agents/{key}/resources/{conversation_id}/event-runs",
        response_model=AgentRun,
        status_code=201,
    )
    def create_event_run(key: AgentKey, conversation_id: UUID, value: EventRunCreate):
        return repository.create_event_run(
            key, conversation_id, value.id, value.origin_run_id, value.event
        )

    @router.get("/agents/{key}/runs/{run_id}", response_model=AgentRun)
    def get_run(key: AgentKey, run_id: UUID):
        return with_draft(key, repository.get_run(key, run_id))

    @router.put("/agents/{key}/runs/{run_id}/draft", status_code=204)
    def update_run_draft(key: AgentKey, run_id: UUID, value: RunDraftUpdate):
        run = repository.get_run(key, run_id)
        if run.status != "running":
            clear_draft(key, run_id)
            raise StorageError("This agent run is not running.", 409)
        if run.owner_id != value.worker_id:
            raise StorageError("This agent run is not owned by this worker.", 409)
        with draft_lock:
            if value.content:
                if len(run_drafts) >= 128 and (key, run_id) not in run_drafts:
                    run_drafts.pop(next(iter(run_drafts)))
                run_drafts[(key, run_id)] = value.content
            else:
                run_drafts.pop((key, run_id), None)

    @router.put("/agents/{key}/runs/{run_id}", response_model=AgentRun)
    def update_run(key: AgentKey, run_id: UUID, value: RunUpdate):
        state = AgentRunState.model_validate(
            value.model_dump(include=set(AgentRunState.model_fields))
        )
        updated = repository.update_run(
            key, run_id, state, value.expected_version, value.worker_id
        )
        if updated.status != "running":
            clear_draft(key, run_id)
        return updated

    @router.post("/agents/{key}/runs/{run_id}/cancel", response_model=AgentRun)
    def cancel_run(key: AgentKey, run_id: UUID):
        clear_draft(key, run_id)
        return repository.cancel_run(key, run_id)

    @router.post("/agents/{key}/runs/{run_id}/messages", response_model=Resource)
    def deliver_run_message(key: AgentKey, run_id: UUID, value: RunDelivery):
        resource = repository.deliver_run_message(
            key, run_id, value.worker_id, value.message, value.image_job_id
        )
        clear_draft(key, run_id)
        return resource

    @router.post(
        "/agents/{key}/runs/{run_id}/image-jobs",
        response_model=ImageJob,
        status_code=201,
    )
    def create_image_job(key: AgentKey, run_id: UUID, value: ImageJobCreate):
        return repository.create_image_job(
            key, run_id, value.id, value.prompt, value.request_id, value.worker_id
        )

    @router.get("/image-jobs", response_model=list[ImageJob])
    def active_image_jobs():
        return repository.active_image_jobs()

    @router.get(
        "/agents/{key}/resources/{conversation_id}/image-jobs",
        response_model=list[ImageJob],
    )
    def conversation_image_jobs(key: AgentKey, conversation_id: UUID):
        return repository.conversation_image_jobs(key, conversation_id)

    @router.get(
        "/agents/{key}/resources/{conversation_id}/task-status",
        response_model=ConversationTaskPage,
    )
    def conversation_task_status(
        key: AgentKey, conversation_id: UUID, offset: Annotated[int, Query(ge=0)] = 0
    ):
        return repository.conversation_task_status(key, conversation_id, offset)

    @router.get("/agents/{key}/image-jobs/{job_id}", response_model=ImageJob)
    def get_image_job(key: AgentKey, job_id: UUID):
        return repository.get_image_job(key, job_id)

    @router.put("/agents/{key}/image-jobs/{job_id}", response_model=ImageJob)
    def update_image_job(key: AgentKey, job_id: UUID, value: ImageJobUpdate):
        state = ImageJobState.model_validate(
            value.model_dump(exclude={"expected_version", "worker_id"})
        )
        return repository.update_image_job(
            key, job_id, state, value.expected_version, value.worker_id
        )

    @router.post("/agents/{key}/image-jobs/{job_id}/cancel", response_model=ImageJob)
    def cancel_image_job(key: AgentKey, job_id: UUID):
        return repository.cancel_image_job(key, job_id)

    @router.post(
        "/agents/{key}/image-jobs/{job_id}/upload/resolve", response_model=ImageJob
    )
    def resolve_image_upload(key: AgentKey, job_id: UUID):
        return files.resolve_image_upload(key, job_id)

    @router.get("/pet-jobs", response_model=list[PetJob])
    def active_pet_jobs():
        return repository.active_pet_jobs()

    @router.get("/agents/{key}/pets", response_model=list[PetJob])
    def list_pets(key: AgentKey):
        return repository.list_pets(key)

    @router.post("/agents/{key}/pets", response_model=PetJob, status_code=201)
    def create_pet(key: AgentKey, value: PetCreate):
        return repository.create_pet(key, value.id, value.prompt, value.reference_mode)

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
        kinds: Annotated[list[ResourceKind] | None, Query()] = None,
    ):
        return repository.list_resources(
            key, kind, q, trash, limit, offset, kinds=kinds
        )

    @router.post("/agents/{key}/resources", response_model=Resource, status_code=201)
    def create_resource(
        key: AgentKey,
        value: ResourceInput,
        preserve_memory_snapshot: bool = False,
    ):
        return repository.create_resource(
            key, value, preserve_memory_snapshot=preserve_memory_snapshot
        )

    @router.post(
        "/agents/{key}/resources/{conversation_id}/messages/{message_id}/document",
        response_model=Resource,
    )
    def save_chat_result(
        key: AgentKey,
        conversation_id: UUID,
        message_id: UUID,
        value: ChatResultInput,
    ):
        return repository.save_chat_result(
            key, conversation_id, message_id, value.title
        )

    @router.get("/agents/{key}/resources/{resource_id}", response_model=Resource)
    def get_resource(key: AgentKey, resource_id: UUID):
        return repository.get_resource(key, resource_id)

    @router.put("/agents/{key}/resources/{resource_id}", response_model=Resource)
    def update_resource(
        key: AgentKey,
        resource_id: UUID,
        value: ResourceUpdate,
        preserve_memory_snapshot: bool = False,
    ):
        fields = value.model_dump(exclude={"expected_version"})
        for name in {
            "pinned",
            "compacted_summary",
            "compacted_through_message_id",
            "compacted_at",
            "memory_extracted_through_message_id",
            "core_memories_snapshot",
            "core_memories_refreshed_at",
        } - value.model_fields_set:
            # Leave unset fields out so saved pin, compaction, and memory state are kept.
            fields.pop(name, None)
        content = resource_input_adapter.validate_python(fields)
        return repository.update_resource(
            key,
            resource_id,
            content,
            value.expected_version,
            preserve_memory_snapshot=preserve_memory_snapshot,
        )

    @router.delete("/agents/{key}/resources/{resource_id}", response_model=Resource)
    def trash_resource(
        key: AgentKey,
        resource_id: UUID,
        expected_version: Annotated[int, Query(ge=1)],
        preserve_memory_snapshot: bool = False,
    ):
        return repository.trash_resource(
            key,
            resource_id,
            expected_version,
            preserve_memory_snapshot=preserve_memory_snapshot,
        )

    @router.post(
        "/agents/{key}/resources/{resource_id}/restore", response_model=Resource
    )
    def restore_resource(
        key: AgentKey,
        resource_id: UUID,
        value: VersionInput,
        preserve_memory_snapshot: bool = False,
    ):
        return repository.restore_resource(
            key,
            resource_id,
            value.expected_version,
            preserve_memory_snapshot=preserve_memory_snapshot,
        )

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
        key: AgentKey,
        resource_id: UUID,
        version: int,
        value: VersionInput,
        preserve_memory_snapshot: bool = False,
    ):
        return repository.restore_version(
            key,
            resource_id,
            version,
            value.expected_version,
            preserve_memory_snapshot=preserve_memory_snapshot,
        )

    @router.post(
        "/agents/{key}/resources/{resource_id}/messages", response_model=Resource
    )
    def append_message(key: AgentKey, resource_id: UUID, value: MessageAppend):
        return repository.append_message(
            key, resource_id, value.message, value.expected_version
        )

    @router.put(
        "/agents/{key}/resources/{resource_id}/compaction", response_model=Resource
    )
    def update_conversation_compaction(
        key: AgentKey, resource_id: UUID, value: ConversationCompactionUpdate
    ):
        return repository.update_conversation_compaction(key, resource_id, value)

    @router.put(
        "/agents/{key}/resources/{resource_id}/memory-state", response_model=Resource
    )
    def update_conversation_memory_state(
        key: AgentKey, resource_id: UUID, value: ConversationMemoryStateUpdate
    ):
        return repository.update_conversation_memory_state(key, resource_id, value)

    @router.get("/agents/{key}/assets", response_model=list[Asset])
    def list_assets(
        key: AgentKey,
        images: bool = False,
        trash: bool = False,
        q: str = "",
        limit: Annotated[int | None, Query(ge=1, le=200)] = None,
        offset: Annotated[int, Query(ge=0)] = 0,
    ):
        return repository.list_assets(key, images, trash, q, limit, offset)

    @router.post("/agents/{key}/assets", response_model=Asset, status_code=201)
    def upload_asset(
        key: AgentKey,
        file: Annotated[UploadFile, File()],
        title: Annotated[str | None, Form()] = None,
        tags: Annotated[str, Form()] = "[]",
        sources: Annotated[str, Form()] = "[]",
        asset_id: Annotated[UUID | None, Form()] = None,
        image_job_id: Annotated[UUID | None, Form()] = None,
        folder_id: Annotated[UUID | None, Form()] = None,
    ):
        try:
            metadata = AssetMetadata(
                title=title or safe_filename(file.filename),
                tags=json.loads(tags),
                sources=json.loads(sources),
                folder_id=folder_id,
            )
        except (json.JSONDecodeError, ValidationError) as error:
            raise StorageError(
                "Upload title, tags or sources are invalid", 422
            ) from error
        return files.upload(
            key, file.file, file.filename, metadata, asset_id, image_job_id
        )

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
            },
        )

    @router.put("/agents/{key}/assets/{asset_id}", response_model=Asset)
    def update_asset(key: AgentKey, asset_id: UUID, value: AssetUpdate):
        fields = value.model_dump(exclude={"expected_version"})
        for name in {"pinned", "folder_id"} - value.model_fields_set:
            # Leave it unset so the saved value is kept.
            del fields[name]
        metadata = AssetMetadata.model_validate(fields)
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

    @router.get("/mcp-servers", response_model=list[McpServer])
    def list_mcp_servers():
        return repository.list_mcp_servers()

    @router.post("/mcp-servers", response_model=McpServer, status_code=201)
    def create_mcp_server(value: McpServerInput):
        return repository.create_mcp_server(value)

    @router.get("/mcp-servers/{server_id}", response_model=McpServer)
    def get_mcp_server(server_id: UUID):
        return repository.get_mcp_server(server_id)

    @router.put("/mcp-servers/{server_id}", response_model=McpServer)
    def update_mcp_server(server_id: UUID, value: McpServerUpdate):
        return repository.update_mcp_server(server_id, value)

    @router.delete("/mcp-servers/{server_id}", status_code=204)
    def delete_mcp_server(server_id: UUID):
        repository.delete_mcp_server(server_id)

    @router.post("/mcp-oauth-states", response_model=McpOAuthState, status_code=201)
    def save_mcp_oauth_state(value: McpOAuthState):
        return repository.save_mcp_oauth_state(value)

    @router.post("/mcp-oauth-states/{state}/consume", response_model=McpOAuthState)
    def consume_mcp_oauth_state(state: str):
        return repository.consume_mcp_oauth_state(state)

    @router.get("/t3-watches", response_model=list[T3ThreadWatch])
    def list_t3_watches(
        status: Literal["watching", "delivered", "dropped"] | None = None,
    ):
        return repository.list_t3_watches(status)

    @router.post("/t3-watches", response_model=T3ThreadWatch, status_code=201)
    def upsert_t3_watch(value: T3ThreadWatchInput):
        return repository.upsert_t3_watch(value)

    @router.put("/t3-watches/{watch_id}", response_model=T3ThreadWatch)
    def update_t3_watch(watch_id: UUID, value: T3ThreadWatchUpdate):
        return repository.update_t3_watch(watch_id, value)

    @router.get("/group-chat/rooms/default", response_model=GroupChatRoom)
    def default_group_chat_room():
        return repository.get_or_create_default_group_chat_room()

    @router.get("/group-chat/rooms/{room_id}", response_model=GroupChatRoom)
    def get_group_chat_room(room_id: UUID):
        return repository.get_group_chat_room(room_id)

    @router.put("/group-chat/rooms/{room_id}", response_model=GroupChatRoom)
    def update_group_chat_room(room_id: UUID, value: GroupChatRoomUpdate):
        return repository.update_group_chat_room(room_id, value)

    @router.get(
        "/group-chat/rooms/{room_id}/participants",
        response_model=list[GroupChatParticipant],
    )
    def list_group_chat_participants(room_id: UUID):
        return repository.list_group_chat_participants(room_id)

    @router.post(
        "/group-chat/rooms/{room_id}/participants",
        response_model=GroupChatParticipant,
        status_code=201,
    )
    def create_group_chat_participant(room_id: UUID, value: GroupChatParticipantInput):
        return repository.create_group_chat_participant(room_id, value)

    @router.put(
        "/group-chat/rooms/{room_id}/participants/{participant_id}",
        response_model=GroupChatParticipant,
    )
    def update_group_chat_participant(
        room_id: UUID, participant_id: UUID, value: GroupChatParticipantUpdate
    ):
        return repository.update_group_chat_participant(room_id, participant_id, value)

    @router.post(
        "/group-chat/rooms/{room_id}/participants/{participant_id}/touch",
        response_model=GroupChatParticipant,
    )
    def touch_group_chat_participant(
        room_id: UUID, participant_id: UUID, value: GroupChatParticipantTouch
    ):
        return repository.touch_group_chat_participant(room_id, participant_id, value)

    @router.post(
        "/group-chat/participants/authenticate",
        response_model=GroupChatParticipant,
    )
    def authenticate_group_chat_participant(value: GroupChatTokenAuth):
        return repository.authenticate_group_chat_token(value.token_hash)

    @router.get(
        "/group-chat/rooms/{room_id}/messages", response_model=GroupChatMessagesPage
    )
    def list_group_chat_messages(
        room_id: UUID,
        after_seq: Annotated[int | None, Query(ge=0)] = None,
        before_seq: Annotated[int | None, Query(ge=1)] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ):
        return repository.list_group_chat_messages(
            room_id, after_seq=after_seq, before_seq=before_seq, limit=limit
        )

    @router.post(
        "/group-chat/rooms/{room_id}/messages",
        response_model=GroupChatMessageWriteResult,
        status_code=201,
    )
    def create_group_chat_message(room_id: UUID, value: GroupChatMessageInput):
        return repository.create_group_chat_message(room_id, value)

    @router.get(
        "/group-chat/rooms/{room_id}/subscriptions",
        response_model=list[GroupChatEventSubscription],
    )
    def list_group_chat_subscriptions(
        room_id: UUID,
        participant_id: UUID | None = None,
        status: GroupChatSubscriptionStatus | None = None,
    ):
        return repository.list_group_chat_subscriptions(
            room_id, participant_id=participant_id, status=status
        )

    @router.post(
        "/group-chat/rooms/{room_id}/subscriptions",
        response_model=GroupChatEventSubscription,
        status_code=201,
    )
    def create_group_chat_subscription(
        room_id: UUID, value: GroupChatEventSubscriptionInput
    ):
        return repository.create_group_chat_subscription(room_id, value)

    @router.get(
        "/group-chat/subscriptions/{subscription_id}",
        response_model=GroupChatEventSubscription,
    )
    def get_group_chat_subscription(subscription_id: UUID):
        return repository.get_group_chat_subscription(subscription_id)

    @router.put(
        "/group-chat/subscriptions/{subscription_id}",
        response_model=GroupChatEventSubscription,
    )
    def update_group_chat_subscription(
        subscription_id: UUID, value: GroupChatEventSubscriptionUpdate
    ):
        return repository.update_group_chat_subscription(subscription_id, value)

    @router.delete("/group-chat/subscriptions/{subscription_id}", status_code=204)
    def revoke_group_chat_subscription(
        subscription_id: UUID, participant_id: UUID | None = None
    ):
        repository.revoke_group_chat_subscription(
            subscription_id, participant_id=participant_id
        )

    @router.get(
        "/group-chat/webhook-deliveries",
        response_model=list[GroupChatWebhookDelivery],
    )
    def list_group_chat_webhook_deliveries(
        status: GroupChatDeliveryStatus | None = None,
        due_before: datetime | None = None,
        room_id: UUID | None = None,
        participant_id: UUID | None = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 100,
    ):
        return repository.list_group_chat_webhook_deliveries(
            status=status,
            due_before=due_before,
            room_id=room_id,
            participant_id=participant_id,
            limit=limit,
        )

    @router.put(
        "/group-chat/webhook-deliveries/{delivery_id}",
        response_model=GroupChatWebhookDelivery,
    )
    def update_group_chat_webhook_delivery(
        delivery_id: UUID, value: GroupChatWebhookDeliveryUpdate
    ):
        return repository.update_group_chat_webhook_delivery(delivery_id, value)

    @router.post(
        "/group-chat/oauth/clients",
        response_model=GroupChatOAuthClient,
        status_code=201,
    )
    def save_group_chat_oauth_client(value: GroupChatOAuthClientInput):
        return repository.save_group_chat_oauth_client(value)

    @router.get(
        "/group-chat/oauth/clients/{client_id:path}",
        response_model=GroupChatOAuthClient,
    )
    def get_group_chat_oauth_client(client_id: str):
        return repository.get_group_chat_oauth_client(client_id)

    @router.post(
        "/group-chat/oauth/codes",
        response_model=GroupChatOAuthCode,
        status_code=201,
    )
    def save_group_chat_oauth_code(value: GroupChatOAuthCodeInput):
        return repository.save_group_chat_oauth_code(value)

    @router.post(
        "/group-chat/oauth/codes/consume",
        response_model=GroupChatOAuthCode,
    )
    def consume_group_chat_oauth_code(value: GroupChatOAuthCodeConsume):
        return repository.consume_group_chat_oauth_code(value.code_hash)

    @router.post(
        "/group-chat/oauth/tokens",
        response_model=GroupChatOAuthToken,
        status_code=201,
    )
    def create_group_chat_oauth_token(value: GroupChatOAuthTokenInput):
        return repository.create_group_chat_oauth_token(value)

    @router.post(
        "/group-chat/oauth/tokens/refresh",
        response_model=GroupChatOAuthToken,
    )
    def refresh_group_chat_oauth_token(value: GroupChatOAuthTokenRefresh):
        return repository.refresh_group_chat_oauth_token(value)

    # ── Native Work storage routes ───────────────────────────────────────────

    @router.get("/work/projects", response_model=list[WorkProject])
    def list_work_projects():
        return repository.list_work_projects()

    @router.post("/work/projects", response_model=WorkProject, status_code=201)
    def create_work_project(value: WorkProjectInput):
        return repository.create_work_project(value)

    @router.get("/work/projects/{project_id}", response_model=WorkProject)
    def get_work_project(project_id: UUID):
        return repository.get_work_project(project_id)

    @router.put("/work/projects/{project_id}", response_model=WorkProject)
    def update_work_project(project_id: UUID, value: WorkProjectUpdate):
        return repository.update_work_project(project_id, value)

    @router.delete("/work/projects/{project_id}", status_code=204)
    def delete_work_project(project_id: UUID, force: bool = False):
        repository.delete_work_project(project_id, force=force)

    @router.get("/work/threads", response_model=list[WorkThread])
    def list_work_threads(
        bucket: WorkBucket | None = None,
        project_id: UUID | None = None,
        q: str = "",
        limit: Annotated[int, Query(ge=1, le=500)] = 200,
    ):
        return repository.list_work_threads(
            bucket=bucket, project_id=project_id, query=q, limit=limit
        )

    @router.post("/work/threads", response_model=WorkThread, status_code=201)
    def create_work_thread(value: WorkThreadInput):
        return repository.create_work_thread(value)

    @router.post("/work/threads/wake-snoozed", response_model=list[WorkThread])
    def wake_snoozed_work_threads():
        return repository.wake_snoozed_work_threads()

    @router.get("/work/threads/{thread_id}", response_model=WorkThread)
    def get_work_thread(thread_id: UUID):
        return repository.get_work_thread(thread_id)

    @router.put("/work/threads/{thread_id}", response_model=WorkThread)
    def update_work_thread(thread_id: UUID, value: WorkThreadUpdate):
        return repository.update_work_thread(thread_id, value)

    @router.post("/work/threads/{thread_id}/organize", response_model=WorkThread)
    def organize_work_thread(thread_id: UUID, value: WorkThreadOrganize):
        return repository.organize_work_thread(thread_id, value)

    @router.delete("/work/threads/{thread_id}", status_code=204)
    def delete_work_thread(thread_id: UUID):
        repository.delete_work_thread(thread_id)

    @router.post(
        "/work/threads/{thread_id}/pending-requests", response_model=WorkThread
    )
    def upsert_work_pending_request(thread_id: UUID, value: WorkPendingRequest):
        return repository.upsert_work_pending_request(thread_id, value)

    @router.post(
        "/work/threads/{thread_id}/pending-requests/{request_id}/respond",
        response_model=WorkThread,
    )
    def respond_work_pending_request(
        thread_id: UUID, request_id: str, value: WorkPendingRequestResponse
    ):
        return repository.respond_work_pending_request(thread_id, request_id, value)

    @router.get("/work/runs", response_model=list[WorkRun])
    def list_work_runs(
        thread_id: UUID | None = None,
        active_only: bool = False,
    ):
        return repository.list_work_runs(thread_id=thread_id, active_only=active_only)

    @router.get("/work/threads/{thread_id}/runs", response_model=list[WorkRun])
    def list_thread_work_runs(thread_id: UUID):
        return repository.list_work_runs(thread_id=thread_id)

    @router.post(
        "/work/threads/{thread_id}/runs", response_model=WorkRun, status_code=201
    )
    def create_work_run(thread_id: UUID, value: WorkRunInput):
        return repository.create_work_run(thread_id, value)

    @router.get("/work/runs/{run_id}", response_model=WorkRun)
    def get_work_run(run_id: UUID):
        return repository.get_work_run(run_id)

    @router.put("/work/runs/{run_id}", response_model=WorkRun)
    def update_work_run(run_id: UUID, value: WorkRunUpdate):
        return repository.update_work_run(run_id, value)

    @router.post("/work/runs/{run_id}/cancel", response_model=WorkRun)
    def cancel_work_run(run_id: UUID):
        return repository.cancel_work_run(run_id)

    @router.get(
        "/work/threads/{thread_id}/activity", response_model=list[WorkActivityItem]
    )
    def list_work_activity(
        thread_id: UUID,
        after_position: int = -1,
        limit: Annotated[int, Query(ge=1, le=500)] = 200,
    ):
        return repository.list_work_activity(
            thread_id, after_position=after_position, limit=limit
        )

    @router.post(
        "/work/threads/{thread_id}/activity",
        response_model=list[WorkActivityItem],
        status_code=201,
    )
    def append_work_activity(thread_id: UUID, items: list[WorkActivityInput]):
        return repository.append_work_activity(thread_id, items)

    @router.get("/profile", response_model=UserProfile)
    def get_user_profile():
        return repository.get_user_profile()

    @router.put("/profile", response_model=UserProfile)
    def update_user_profile(value: UserProfileUpdate):
        return repository.update_user_profile(value)

    @router.get("/work/harnesses", response_model=list[WorkHarnessConfig])
    def list_work_harnesses():
        return repository.list_work_harnesses()

    @router.get("/work/harnesses/{harness}", response_model=WorkHarnessConfig | None)
    def get_work_harness(harness: WorkHarnessId):
        return repository.get_work_harness(harness)

    @router.put("/work/harnesses/{harness}", response_model=WorkHarnessConfig)
    def put_work_harness(harness: WorkHarnessId, value: WorkHarnessConfigInput):
        return repository.put_work_harness(harness, value)

    @router.delete("/work/harnesses/{harness}", status_code=204)
    def delete_work_harness(harness: WorkHarnessId):
        repository.delete_work_harness(harness)

    return router
