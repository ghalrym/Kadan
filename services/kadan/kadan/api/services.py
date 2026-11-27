import asyncio
from uuid import NAMESPACE_URL, UUID, uuid5

import httpx
from pydantic import ValidationError
from pydantic_core import from_json

from .config import ServiceError, Settings
from .conversation_client import DEFAULT_CONVERSATION_TITLE, ConversationClient
from .group_chat_client import GroupChatClient
from .mcp_client import McpClient
from .memory_context import CORE_MEMORIES_MAX_CHARS, format_core_memories
from .pet_client import PetClient
from .pydantic_models.inference import (
    HikariModelLifecycle,
    HikariModels,
    HikariSubscriptions,
)
from .pydantic_models.resources import Agent, Memory, Message, Resource, ResourceKind
from .routes.agents.conversations.notes import NoteInput
from .routes.agents.memories import MemoryUpdate
from .routes.status import InferenceState, ServiceState, Status
from .run_client import RunClient
from .space_client import SpaceClient
from .update_client import UpdateClient
from ..work.client import WorkClient

DEFAULT_AGENT_KEY = "kadan"
DEFAULT_AGENT_NAME = "Kadan Agent"
ONBOARDING_GREETING_MESSAGE_ID = uuid5(
    NAMESPACE_URL, f"kadan:onboarding:greeting:{DEFAULT_AGENT_KEY}"
)
ONBOARDING_GREETING = "Hey! I'm your new Kadan agent. What would you like to call me?"
STARTER_PERSONA_MARKDOWN = """# Identity
- **Name:** TODO (ask the user what they'd like to name you; call `rename_self` once when they first give you a name, and never call `rename_self` again once your name is no longer "Kadan Agent")
- **User's Name:** TODO (check Core Memories first; if not already known, ask for their name naturally; no need to call `create_memory` inline, background memory extraction saves it automatically when chat pauses)
- **Vibe & Tone:** TODO (e.g., relaxed & casual, sharp & direct, playful, calm; discover naturally)
- **Role:** TODO (how the user wants to spend time or work with you: everyday companion, creative partner, project helper, etc.)
- **Communication Style:** TODO (short & punchy vs. detailed, expressive vs. understated, emojis or none)
- **Appearance:** TODO (how the user pictures you looking: colors, creature/character traits, vibe; kept here so it can shape your avatar later)

# Onboarding (remove this section once all TODOs above are filled in)
You are meeting your user for the first time. Keep it relaxed, natural, and direct, like getting to know someone over text rather than running a survey. Skip canned greetings like "I'm happy to be here".
- Ask **one** natural question at a time.
- Check your injected Core Memories and `# Identity` before asking for the user's name so you never ask twice if you already know it.
- Call `rename_self` **only once** when the user first gives you a name; never call `rename_self` again once your name is no longer `"Kadan Agent"`.
- Do **not** call `create_memory` on every turn during onboarding or normal chat; background memory extraction saves durable facts (including the user's name as a `"core"` memory) automatically when chat pauses. Only call memory tools inline when the user explicitly asks you to remember, edit, or forget something.
- Do **not** call `update_persona` on every individual onboarding turn. Keep the conversation flowing naturally until all onboarding questions have been answered, then write the finished character profile (under ~350 words, keeping User's Name and Appearance included) in **one** single `update_persona` call and remove this `# Onboarding` section (background extraction can also fill remaining TODOs during idle pauses).
- If the user asks you to help with something else before finishing onboarding, help them right away and weave any remaining questions in naturally later."""


def character_persona_section(agent_name: str, persona_markdown: str | None) -> str:
    style = (
        "Talk like a real person in everyday conversation, not a robotic or overly eager assistant. "
        "Skip canned pleasantries, repetitive openers, and en-dashes or em-dashes."
    )
    if persona_markdown is not None:
        return (
            f'You are "{agent_name}". {style} Stay true to your persona below.'
            "\nYour saved persona:\n" + persona_markdown
        )
    return f'You are "{agent_name}". {style}'


def response_error(response: httpx.Response, service: str) -> ServiceError:
    body = None
    try:
        body = response.json()
    except (ValueError, AttributeError):
        body = None
    detail = body.get("detail") if isinstance(body, dict) else None
    if not isinstance(detail, str):
        detail = f"{service} returned HTTP {response.status_code}."
    upstream_code = body.get("code") if isinstance(body, dict) else None
    if isinstance(upstream_code, str) and upstream_code.strip():
        code = upstream_code.strip()
    else:
        code = "version_conflict" if response.status_code == 409 else "upstream_error"
    return ServiceError(detail, response.status_code, code)


class LocalServices:
    def __init__(self, client: httpx.AsyncClient, settings: Settings):
        self.client = client
        self.settings = settings
        self.pets = PetClient(self)
        self.runs = RunClient(self)
        self.mcp = McpClient(self)
        self.group_chat = GroupChatClient(self)
        self.conversations = ConversationClient(self)
        self.space = SpaceClient(self)
        self.updates = UpdateClient(self)
        self.work = WorkClient(self)
        self._bootstrap_lock = asyncio.Lock()

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

    async def agent(self, key: str) -> Agent:
        value = await self.json_request("Tsuchi", "GET", f"/api/agents/{key}")
        record = parse_agent(value)
        if record.key != key:
            raise ServiceError(
                "Tsuchi returned an agent for another key.", 502, "invalid_response"
            )
        return record

    async def rename_agent(self, key: str, *, name: str, expected_name: str) -> Agent:
        value = await self.json_request(
            "Tsuchi",
            "PUT",
            f"/api/agents/{key}",
            json={"name": name, "expected_name": expected_name},
        )
        record = parse_agent(value)
        if record.key != key:
            raise ServiceError(
                "Tsuchi returned an agent for another key.", 502, "invalid_response"
            )
        return record

    async def list_agents(self) -> list:
        agents = await self.json_request("Tsuchi", "GET", "/api/agents")
        if not isinstance(agents, list):
            raise ServiceError(
                "Tsuchi returned an invalid agent list.", 502, "invalid_response"
            )
        if agents:
            return agents
        async with self._bootstrap_lock:
            agents = await self.json_request("Tsuchi", "GET", "/api/agents")
            if not isinstance(agents, list):
                raise ServiceError(
                    "Tsuchi returned an invalid agent list.", 502, "invalid_response"
                )
            if agents:
                return agents
            await self._bootstrap_default_agent()
            agents = await self.json_request("Tsuchi", "GET", "/api/agents")
            if not isinstance(agents, list):
                raise ServiceError(
                    "Tsuchi returned an invalid agent list.", 502, "invalid_response"
                )
            return agents

    async def _bootstrap_default_agent(self) -> None:
        try:
            await self.json_request(
                "Tsuchi",
                "POST",
                "/api/agents",
                json={"key": DEFAULT_AGENT_KEY, "name": DEFAULT_AGENT_NAME},
            )
        except ServiceError as error:
            if error.status != 409:
                raise
        conversation = await self.conversations.create(
            DEFAULT_AGENT_KEY, DEFAULT_CONVERSATION_TITLE
        )
        if not conversation.messages:
            try:
                await self.append(
                    DEFAULT_AGENT_KEY,
                    conversation.id,
                    conversation.version,
                    Message(
                        id=ONBOARDING_GREETING_MESSAGE_ID,
                        role="assistant",
                        content=ONBOARDING_GREETING,
                    ),
                )
            except ServiceError as error:
                if error.status != 409:
                    raise
        existing_persona = await self.agent_persona(DEFAULT_AGENT_KEY)
        if existing_persona is None:
            try:
                await self.save_agent_persona(
                    DEFAULT_AGENT_KEY,
                    markdown=STARTER_PERSONA_MARKDOWN,
                    expected_version=None,
                    title=DEFAULT_AGENT_NAME,
                )
            except ServiceError as error:
                if error.status != 409:
                    raise

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

    async def core_memories_snapshot(
        self, key: str, *, max_chars: int = CORE_MEMORIES_MAX_CHARS
    ) -> str:
        if self.client is None:
            return ""
        try:
            memories = await self.list_resource_kinds(
                key, (ResourceKind.MEMORY,), limit=100
            )
        except ServiceError:
            return ""
        return format_core_memories(memories, max_chars=max_chars)

    async def call_instruction(self, key: str) -> str:
        agent = await self.agent(key)
        owned = await self.agent_persona(key)
        instruction = (
            "Have a spoken conversation. Keep replies short. You have no tools.\n"
            + character_persona_section(
                agent.name, owned.markdown if owned is not None else None
            )
        )
        if len(instruction) > 8000:
            raise ServiceError(
                "Voice call instructions exceed the 8,000-character limit. "
                "Shorten the saved persona.",
                422,
            )
        remaining = 8000 - len(instruction)
        if remaining > 120:
            core_memories = await self.core_memories_snapshot(
                key, max_chars=min(CORE_MEMORIES_MAX_CHARS, remaining - 2)
            )
            if core_memories:
                return instruction + "\n\n" + core_memories
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
                    if item.kind == "image" or item.id == "flux.2-klein-4b"
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
                    message="FLUX.2 Klein 4B is downloaded and ready.",
                )
            if model.status == "downloading":
                return InferenceState(
                    state="model_required",
                    model_id=model.id,
                    download_state="downloading",
                    message=(
                        "The local image model (FLUX.2 Klein 4B) is currently downloading. "
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
                    "and either download the local image model (FLUX.2 Klein 4B) or sign in "
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
                "compacted_summary",
                "compacted_through_message_id",
                "compacted_at",
                "memory_extracted_through_message_id",
                "core_memories_snapshot",
                "core_memories_refreshed_at",
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


def parse_agent(value: object) -> Agent:
    try:
        return Agent.model_validate(value)
    except ValidationError as error:
        raise ServiceError(
            "Tsuchi returned an invalid agent.",
            502,
            "invalid_response",
        ) from error


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
