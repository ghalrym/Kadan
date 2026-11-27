import asyncio
from uuid import UUID

import httpx
from pydantic import ValidationError
from pydantic_core import from_json

from .config import ServiceError, Settings
from .conversation_client import ConversationClient
from .mcp_client import McpClient
from .pet_client import PetClient
from .pydantic_models.inference import (
    HikariModelLifecycle,
    HikariModels,
    HikariSubscriptions,
)
from .pydantic_models.resources import Memory, Message, Resource, ResourceKind
from .routes.agents.conversations.notes import NoteInput
from .routes.agents.memories import MemoryUpdate
from .routes.status import InferenceState, ServiceState, Status
from .run_client import RunClient
from .space_client import SpaceClient


def response_error(response: httpx.Response, service: str) -> ServiceError:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if not isinstance(detail, str):
        detail = f"{service} returned HTTP {response.status_code}."
    code = "version_conflict" if response.status_code == 409 else "upstream_error"
    return ServiceError(detail, response.status_code, code)


class LocalServices:
    def __init__(self, client: httpx.AsyncClient, settings: Settings):
        self.client = client
        self.settings = settings
        self.pets = PetClient(self)
        self.runs = RunClient(self)
        self.mcp = McpClient(self)
        self.conversations = ConversationClient(self)
        self.space = SpaceClient(self)

    async def json_request(
        self,
        service: str,
        method: str,
        path: str,
        *,
        timeout: float = 15,
        **kwargs,
    ):
        base = (
            self.settings.tsuchi_url
            if service == "Tsuchi"
            else self.settings.hikari_url
        )
        try:
            response = await self.client.request(
                method, base + path, timeout=httpx.Timeout(timeout, connect=3), **kwargs
            )
        except httpx.RequestError as error:
            raise ServiceError(
                f"{service} is unavailable. Check that its local service is running."
            ) from error
        if not response.is_success:
            raise response_error(response, service)
        if response.status_code == 204:
            return None
        try:
            return from_json(response.content, cache_strings="keys")
        except ValueError as error:
            raise ServiceError(
                f"{service} returned an invalid response.", 502, "invalid_response"
            ) from error

    async def resource(self, key: str, resource_id: UUID) -> Resource:
        value = await self.json_request(
            "Tsuchi", "GET", f"/api/agents/{key}/resources/{resource_id}"
        )
        return parse_resource(value)

    async def agent_persona(self, key: str) -> Resource | None:
        value = await self.json_request("Tsuchi", "GET", f"/api/agents/{key}/persona")
        if value is None:
            return None
        persona = parse_resource(value)
        require_active(persona, ResourceKind.PERSONA)
        if persona.agent_key != key:
            raise ServiceError("Tsuchi returned a persona for another agent.", 502)
        return persona

    async def save_agent_persona(
        self,
        key: str,
        *,
        markdown: str,
        expected_version: int | None,
        title: str | None = None,
    ) -> Resource:
        resolved_title = title.strip() if title and title.strip() else None
        if resolved_title is None:
            current = await self.agent_persona(key)
            if current is not None and current.title.strip():
                resolved_title = current.title.strip()
            else:
                agent = await self.json_request("Tsuchi", "GET", f"/api/agents/{key}")
                name = agent.get("name", "").strip() if isinstance(agent, dict) else ""
                resolved_title = name[:200] if name else key[:200]
        value = await self.json_request(
            "Tsuchi",
            "PUT",
            f"/api/agents/{key}/persona",
            json={
                "title": resolved_title,
                "markdown": markdown,
                "expected_version": expected_version,
            },
        )
        persona = parse_resource(value)
        require_active(persona, ResourceKind.PERSONA)
        if persona.agent_key != key:
            raise ServiceError("Tsuchi returned a persona for another agent.", 502)
        return persona

    async def call_instruction(self, key: str) -> str:
        instruction = (
            "You are the user's agent. Have a spoken conversation. "
            "Keep replies short. You have no tools."
        )
        owned = await self.agent_persona(key)
        if owned is not None:
            instruction += "\nYour saved persona:\n" + owned.markdown
        if len(instruction) > 8000:
            raise ServiceError(
                "Voice call instructions exceed the 8,000-character limit. "
                "Shorten the saved persona.",
                422,
            )
        return instruction

    async def list_resource_kinds(
        self,
        key: str,
        kinds: tuple[ResourceKind, ...],
        *,
        q: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> list[Resource]:
        values = await self.json_request(
            "Tsuchi",
            "GET",
            f"/api/agents/{key}/resources",
            params={
                "kinds": [kind.value for kind in kinds],
                "q": q,
                "limit": limit,
                "offset": offset,
                "trash": "false",
            },
        )
        if not isinstance(values, list):
            raise ServiceError(
                "Tsuchi returned an invalid collection.", 502, "invalid_response"
            )
        resources = [parse_resource(value) for value in values]
        if any(
            resource.agent_key != key
            or resource.kind not in kinds
            or resource.deleted_at is not None
            for resource in resources
        ):
            raise ServiceError(
                "Tsuchi returned an invalid collection.", 502, "invalid_response"
            )
        return resources

    async def resource_of_kinds(
        self, key: str, resource_id: UUID, kinds: tuple[ResourceKind, ...]
    ) -> Resource:
        resource = await self.resource(key, resource_id)
        if resource.agent_key != key or resource.id != resource_id:
            raise ServiceError(
                "Tsuchi returned an invalid resource.", 502, "invalid_response"
            )
        if resource.kind not in kinds or resource.deleted_at is not None:
            raise ServiceError("Record not found in this collection.", 404, "not_found")
        return resource

    async def inference_status(self) -> InferenceState:
        try:
            providers = HikariSubscriptions.model_validate(
                await self.json_request("Hikari", "GET", "/v1/subscriptions")
            )
            if providers.capabilities.language != "local":
                # Hikari routes omitted-model chat requests to the user's subscription.
                return InferenceState(
                    state="ready",
                    model_id=None,
                    message=f"Language uses your {providers.capabilities.language.title()} subscription in Hikari.",
                )
            catalog = HikariModels.model_validate(
                await self.json_request("Hikari", "GET", "/v1/models")
            )
            model = next(
                (
                    item
                    for item in catalog.models
                    if catalog.selected_model_id
                    and item.id == catalog.selected_model_id
                ),
                None,
            ) or next(
                (
                    item
                    for item in catalog.models
                    if item.kind == "llm" or item.id.startswith("inkling-2-mini-")
                ),
                None,
            )
            if model is None or not model.inference_available:
                return InferenceState(
                    state="unavailable",
                    model_id=model.id
                    if model is not None
                    else catalog.selected_model_id,
                    message="The Inkling 2 Mini language worker is unavailable. Check Hikari Settings.",
                )
            return InferenceState(
                state="ready" if model.status == "complete" else "model_required",
                model_id=model.id,
                download_state=model.status,
                context_limit=model.context_limit or model.architecture_context_limit,
                message=(
                    "Inkling 2 Mini is downloaded. Hikari loads it when chat starts."
                    if model.status == "complete"
                    else "Inkling 2 Mini requires a user-approved download in Hikari Settings before chat."
                ),
            )
        except (ServiceError, ValidationError) as error:
            message = (
                str(error)
                if isinstance(error, ServiceError)
                else "Hikari returned invalid model status."
            )
            return InferenceState(state="unavailable", message=message)

    async def image_inference_status(self) -> InferenceState:
        try:
            providers = HikariSubscriptions.model_validate(
                await self.json_request("Hikari", "GET", "/v1/subscriptions")
            )
            if providers.capabilities.image != "local":
                return InferenceState(
                    state="ready",
                    model_id=None,
                    message="Image generation uses ChatGPT through your Codex subscription in Hikari.",
                )
            catalog = HikariModels.model_validate(
                await self.json_request("Hikari", "GET", "/v1/models")
            )
            model = next(
                (
                    item
                    for item in catalog.models
                    if item.kind == "image" or item.id.startswith("qwen-image-")
                ),
                None,
            )
            if model is None or not model.inference_available:
                return InferenceState(
                    state="unavailable",
                    model_id=model.id if model is not None else None,
                    message=(
                        "Image generation is currently unavailable. Tell the user to open "
                        "Kadan Server and check Hikari → Settings."
                    ),
                )
            if model.status == "complete":
                return InferenceState(
                    state="ready",
                    model_id=model.id,
                    download_state="complete",
                    message="Qwen-Image 2.1 Turbo is downloaded and ready.",
                )
            if model.status == "downloading":
                return InferenceState(
                    state="model_required",
                    model_id=model.id,
                    download_state="downloading",
                    message=(
                        "The local image model (Qwen-Image 2.1 Turbo) is currently downloading. "
                        "Tell the user to check download progress in Kadan Server under "
                        "Hikari → Settings and try again once it finishes."
                    ),
                )
            return InferenceState(
                state="model_required",
                model_id=model.id,
                download_state=model.status,
                message=(
                    "Image generation is not set up yet because the image model has not been "
                    "downloaded. Tell the user to open Kadan Server, go to Hikari → Settings, "
                    "and either download the local image model (Qwen-Image 2.1 Turbo) or sign in "
                    "to Codex and switch Images to ChatGPT."
                ),
            )
        except (ServiceError, ValidationError):
            return InferenceState(
                state="unavailable",
                message=(
                    "Could not reach Hikari to check image generation. Tell the user to make "
                    "sure Kadan Server is running and check Hikari → Settings."
                ),
            )

    async def storage_status(self) -> ServiceState:
        try:
            value = await self.json_request("Tsuchi", "GET", "/api/health")
            if not isinstance(value, dict) or value.get("status") != "ok":
                raise ServiceError("Tsuchi returned an invalid health response.")
            return ServiceState(
                state="ready",
                message="Tsuchi stores agents, conversations and workspace files.",
            )
        except ServiceError as error:
            return ServiceState(
                state="unavailable",
                message=str(error),
            )

    async def model_state(self, model_id: str | None):
        try:
            value = HikariModelLifecycle.model_validate(
                await self.json_request("Hikari", "GET", "/model-lifecycle")
            )
        except (ServiceError, ValidationError):
            return None
        # A different resident model means the selected one still has to load.
        return value.state if value.model_id == model_id else "unloaded"

    async def status(self) -> Status:
        storage, inference = await asyncio.gather(
            self.storage_status(), self.inference_status()
        )
        if inference.state == "ready" and inference.model_id is not None:
            inference.model_state = await self.model_state(inference.model_id)
        return Status(tsuchi=storage, hikari=inference)

    async def update_memory(
        self, key: str, resource_id: UUID, value: MemoryUpdate
    ) -> Memory:
        current = await self.resource_of_kinds(key, resource_id, (ResourceKind.MEMORY,))
        require_active(current, ResourceKind.MEMORY, value.expected_version)
        content = current.model_dump(
            mode="json",
            exclude={
                "id",
                "agent_key",
                "version",
                "created_at",
                "updated_at",
                "deleted_at",
            },
        )
        content.update(value.model_dump(mode="json"))
        result = parse_resource(
            await self.json_request(
                "Tsuchi",
                "PUT",
                f"/api/agents/{key}/resources/{resource_id}",
                json=content,
            )
        )
        if (
            result.agent_key != key
            or result.id != resource_id
            or result.kind != ResourceKind.MEMORY
            or result.deleted_at is not None
            or result.version != value.expected_version + 1
            or result.title != value.title
            or result.markdown != value.markdown
        ):
            raise ServiceError(
                "Tsuchi did not confirm this memory correction. Check the saved memory before retrying.",
                502,
                "invalid_response",
            )
        return Memory.model_validate(result.model_dump())

    async def append(
        self, key: str, conversation_id: UUID, version: int, message: Message
    ) -> Resource:
        value = await self.json_request(
            "Tsuchi",
            "POST",
            f"/api/agents/{key}/resources/{conversation_id}/messages",
            json={
                "expected_version": version,
                "message": message.model_dump(mode="json"),
            },
        )
        return parse_resource(value)

    async def save_note(
        self, key: str, conversation_id: UUID, value: NoteInput
    ) -> Resource:
        current = await self.resource(key, conversation_id)
        require_active(current, ResourceKind.CONVERSATION, value.expected_version)
        if any(message.id == value.message_id for message in current.messages):
            raise ServiceError(
                "This message is already saved. Reload the conversation.",
                409,
                "duplicate_message",
            )
        message = Message(
            id=value.message_id,
            role="user",
            content=value.content,
            attachment_ids=value.attachment_ids,
        )
        return await self.append(key, conversation_id, value.expected_version, message)


def parse_resource(value: object) -> Resource:
    try:
        return Resource.model_validate(value)
    except ValidationError as error:
        raise ServiceError(
            "Tsuchi returned an invalid resource. Reload to check whether the last change was saved.",
            502,
            "invalid_response",
        ) from error


def require_active(
    resource: Resource, kind: ResourceKind, version: int | None = None
) -> None:
    if resource.deleted_at is not None:
        raise ServiceError(
            "This resource is in Tsuchi's trash. Restore it there before continuing.",
            409,
            "trashed_resource",
        )
    if resource.kind != kind:
        raise ServiceError(
            f"This operation requires a {kind.value}.", 422, "wrong_resource_kind"
        )
    if version is not None and resource.version != version:
        raise ServiceError(
            "This resource changed in Tsuchi. Reload it before saving.",
            409,
            "version_conflict",
        )
