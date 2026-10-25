import asyncio
from uuid import UUID

import httpx
from pydantic import ValidationError
from pydantic_core import from_json

from .config import ServiceError, Settings
from .pet_client import PetClient
from .pydantic_models.inference import (
    HikariModelLifecycle,
    HikariSettings,
)
from .pydantic_models.resources import Message, Resource, ResourceKind
from .routes.agents.conversations.notes import NoteInput
from .routes.agents.resources import PersonaUpdate
from .routes.status import InferenceState, ServiceState, Status
from .run_client import RunClient
from .sandbox_client import SandboxClient

CHAIN_TITLE = "Chain"


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
        self.sandbox = SandboxClient(self)
        self.chain_locks: dict[str, asyncio.Lock] = {}

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

    async def chain(self, key: str) -> Resource:
        # Tsuchi assigns resource IDs, so serialize first use to create one chain.
        async with self.chain_locks.setdefault(key, asyncio.Lock()):
            oldest = None
            offset = 0
            while True:
                rows = await self.json_request(
                    "Tsuchi",
                    "GET",
                    f"/api/agents/{key}/resources",
                    params={
                        "kind": "conversation",
                        "trash": "false",
                        "limit": 200,
                        "offset": offset,
                    },
                )
                for row in rows:
                    candidate = parse_resource(row)
                    if oldest is None or (candidate.created_at, str(candidate.id)) < (
                        oldest.created_at,
                        str(oldest.id),
                    ):
                        oldest = candidate
                if len(rows) < 200:
                    break
                offset += len(rows)
            if oldest is not None:
                return oldest
            value = await self.json_request(
                "Tsuchi",
                "POST",
                f"/api/agents/{key}/resources",
                json={"kind": "conversation", "title": CHAIN_TITLE},
            )
            return parse_resource(value)

    async def inference_status(self) -> InferenceState:
        settings_url = self.settings.hikari_browser_url + "/settings"
        try:
            value = HikariSettings.model_validate(
                await self.json_request("Hikari", "GET", "/v1/settings")
            )
            tier = value.tier
            if tier is None:
                return InferenceState(
                    state="tier_required",
                    message="Choose a tier in Hikari Settings to enable chat. You can save notes now.",
                    settings_url=settings_url,
                )
            selected = next((item for item in value.tiers if item.id == tier), None)
            model = (
                next(
                    (item.model for item in selected.models if item.modality == "llm"),
                    None,
                )
                if selected
                else None
            )
            if model is None:
                return InferenceState(
                    state="unavailable",
                    tier=tier,
                    settings_url=settings_url,
                    message="Hikari has no language model for the selected tier. Check Hikari Settings.",
                )
            lifecycle = await self.model_lifecycle()
            context_limit = None
            context_source = "unknown"
            model_state = None
            if lifecycle is not None:
                model_state = (
                    lifecycle.state if lifecycle.model_id == model.id else "unloaded"
                )
                if lifecycle.state == "ready":
                    if (
                        lifecycle.model_id == model.id
                        and lifecycle.effective_context_limit
                    ):
                        context_limit = lifecycle.effective_context_limit
                        context_source = "effective"
                else:
                    context_limit = (
                        model.context_limit or model.architecture_context_limit
                    )
                    context_source = (
                        "configured"
                        if model.context_limit
                        else (
                            "architecture"
                            if model.architecture_context_limit
                            else "unknown"
                        )
                    )
            status = model.status
            if status != "complete":
                progress = (
                    "is downloading" if status == "downloading" else "is not downloaded"
                )
                if status == "failed":
                    progress = "could not be downloaded"
                return InferenceState(
                    state="model_required",
                    tier=tier,
                    model_id=model.id,
                    download_state=status,
                    context_limit=context_limit,
                    context_source=context_source,
                    model_state=model_state,
                    settings_url=settings_url,
                    message=f"The selected language model {progress}. Check Hikari Settings. You can save notes now.",
                )
            if not model.inference_available:
                return InferenceState(
                    state="unavailable",
                    tier=tier,
                    model_id=model.id,
                    download_state=status,
                    context_limit=context_limit,
                    context_source=context_source,
                    model_state=model_state,
                    settings_url=settings_url,
                    message="The selected language model does not support inference. Check Hikari Settings.",
                )
            return InferenceState(
                state="ready",
                tier=tier,
                model_id=model.id,
                download_state=status,
                context_limit=context_limit,
                context_source=context_source,
                model_state=model_state,
                settings_url=settings_url,
                message="The selected language model is downloaded. Hikari loads it when chat starts.",
            )
        except (ServiceError, ValidationError) as error:
            message = (
                str(error)
                if isinstance(error, ServiceError)
                else "Hikari returned invalid settings."
            )
            return InferenceState(
                state="unavailable", message=message, settings_url=settings_url
            )

    async def storage_status(self) -> ServiceState:
        try:
            value = await self.json_request("Tsuchi", "GET", "/api/health")
            if not isinstance(value, dict) or value.get("status") != "ok":
                raise ServiceError("Tsuchi returned an invalid health response.")
            return ServiceState(
                state="ready",
                message="Tsuchi stores agents, conversations and workspace files.",
                browser_url=self.settings.tsuchi_browser_url,
            )
        except ServiceError as error:
            return ServiceState(
                state="unavailable",
                message=str(error),
                browser_url=self.settings.tsuchi_browser_url,
            )

    async def model_lifecycle(self) -> HikariModelLifecycle | None:
        try:
            value = HikariModelLifecycle.model_validate(
                await self.json_request("Hikari", "GET", "/model-lifecycle")
            )
        except (ServiceError, ValidationError):
            return None
        return value

    async def status(self) -> Status:
        storage, inference = await asyncio.gather(
            self.storage_status(), self.inference_status()
        )
        return Status(tsuchi=storage, hikari=inference)

    async def update_persona(
        self, key: str, resource_id: UUID, value: PersonaUpdate
    ) -> Resource:
        current = await self.resource(key, resource_id)
        require_active(current, ResourceKind.PERSONA, value.expected_version)
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
        result = await self.json_request(
            "Tsuchi", "PUT", f"/api/agents/{key}/resources/{resource_id}", json=content
        )
        return parse_resource(result)

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
        message = Message(id=value.message_id, role="user", content=value.content)
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
