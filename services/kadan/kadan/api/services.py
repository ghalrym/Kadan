import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import aclosing
from uuid import UUID

import httpx
from pydantic import ValidationError

from .config import ServiceError, Settings
from .pet_jobs import PetJobs
from .plugins import PluginRegistry
from .pydantic_models.inference import (
    CompletionMessage,
    HikariModelLifecycle,
    HikariSettings,
    PreparedChat,
)
from .pydantic_models.resources import Message, Resource, ResourceKind
from .routes.agents.conversations.chat import ChatInput
from .routes.agents.conversations.notes import NoteInput
from .routes.agents.resources import PersonaUpdate
from .routes.status import InferenceState, ServiceState, Status
from .tool_loop import INFERENCE_TIMEOUT, ToolTurn, tool_events, tool_history

MAX_PROMPT_MESSAGES = 100
MAX_PROMPT_CHARACTERS = 48000
MAX_REPLY_CHARACTERS = 1000000


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
        self.plugins = PluginRegistry(settings.plugins)
        self.pets = PetJobs(self)

    async def json_request(self, service: str, method: str, path: str, **kwargs):
        base = (
            self.settings.tsuchi_url
            if service == "Tsuchi"
            else self.settings.hikari_url
        )
        try:
            response = await self.client.request(
                method, base + path, timeout=httpx.Timeout(15, connect=3), **kwargs
            )
        except httpx.RequestError as error:
            raise ServiceError(
                f"{service} is unavailable. Check that its local service is running."
            ) from error
        if not response.is_success:
            raise response_error(response, service)
        try:
            return response.json()
        except ValueError as error:
            raise ServiceError(
                f"{service} returned an invalid response.", 502, "invalid_response"
            ) from error

    async def resource(self, key: str, resource_id: UUID) -> Resource:
        value = await self.json_request(
            "Tsuchi", "GET", f"/api/agents/{key}/resources/{resource_id}"
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
                    settings_url=settings_url,
                    message=f"The selected language model {progress}. Check Hikari Settings. You can save notes now.",
                )
            if not model.inference_available:
                return InferenceState(
                    state="unavailable",
                    tier=tier,
                    model_id=model.id,
                    settings_url=settings_url,
                    message="The selected language model does not support inference. Check Hikari Settings.",
                )
            return InferenceState(
                state="ready",
                tier=tier,
                model_id=model.id,
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
        if inference.state == "ready":
            inference.model_state = await self.model_state(inference.model_id)
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

    async def prepare_chat(
        self, key: str, conversation_id: UUID, value: ChatInput
    ) -> PreparedChat:
        current = await self.resource(key, conversation_id)
        require_active(current, ResourceKind.CONVERSATION, value.expected_version)
        if any(message.id == value.message_id for message in current.messages):
            raise ServiceError(
                "This message is already saved. Reload the conversation.",
                409,
                "duplicate_message",
            )
        persona = (
            await self.resource(key, value.persona_id) if value.persona_id else None
        )
        if persona:
            require_active(persona, ResourceKind.PERSONA)
        user = Message(
            id=value.message_id,
            role="user",
            content=value.content,
            resource_ids=[persona.id] if persona else [],
        )
        messages = completion_messages(current, persona, user)
        plugin_ids = self.plugins.select(value.plugin_ids)
        status = await self.inference_status()
        if status.state != "ready":
            raise ServiceError(status.message, 503, status.state)
        return PreparedChat(
            conversation=current,
            user_message=user,
            messages=messages,
            model_id=status.model_id,
            plugin_ids=plugin_ids,
        )

    async def save_chat_user(self, key: str, chat: PreparedChat) -> Resource:
        return await self.append(
            key, chat.conversation.id, chat.conversation.version, chat.user_message
        )

    async def chat_events(
        self, key: str, chat: PreparedChat, saved: Resource
    ) -> AsyncIterator[str]:
        yield event("user_saved", {"conversation": saved.model_dump(mode="json")})
        content: list[str] = []
        length = 0
        finish = None
        done = False
        try:
            turn = ToolTurn(conversation=saved, messages=list(chat.messages))
            async with aclosing(
                tool_events(self, key, chat.model_id, chat.plugin_ids, turn)
            ) as updates:
                async for update in updates:
                    yield event(update.event, update.data)
            saved = turn.conversation
            async with self.client.stream(
                "POST",
                self.settings.hikari_url + "/v1/chat/completions",
                json={
                    "messages": [message.model_dump() for message in turn.messages],
                    "model": chat.model_id,
                    "stream": True,
                    "conversation_id": f"kadan:{key}:{saved.id}",
                },
                timeout=INFERENCE_TIMEOUT,
            ) as response:
                if not response.is_success:
                    await response.aread()
                    raise response_error(response, "Hikari")
                if not response.headers.get("content-type", "").startswith(
                    "text/event-stream"
                ):
                    raise ServiceError(
                        "Hikari did not return a chat stream.", 502, "invalid_stream"
                    )
                async for data in sse_data(response.aiter_lines()):
                    if data == "[DONE]":
                        done = True
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError as error:
                        raise ServiceError(
                            "Hikari returned an invalid chat event.",
                            502,
                            "invalid_stream",
                        ) from error
                    if not isinstance(chunk, dict):
                        raise ServiceError(
                            "Hikari returned an invalid chat event.",
                            502,
                            "invalid_stream",
                        )
                    if "error" in chunk:
                        failure = chunk["error"]
                        message = (
                            failure.get("message")
                            if isinstance(failure, dict)
                            else None
                        )
                        raise ServiceError(
                            message
                            if isinstance(message, str)
                            else "Hikari could not complete this reply.",
                            502,
                            "inference_error",
                        )
                    choices = chunk.get("choices")
                    if (
                        not isinstance(choices, list)
                        or len(choices) != 1
                        or not isinstance(choices[0], dict)
                    ):
                        raise ServiceError(
                            "Hikari returned an invalid chat choice.",
                            502,
                            "invalid_stream",
                        )
                    choice = choices[0]
                    delta = choice.get("delta")
                    if not isinstance(delta, dict) or (
                        "content" in delta and not isinstance(delta["content"], str)
                    ):
                        raise ServiceError(
                            "Hikari returned an invalid text delta.",
                            502,
                            "invalid_stream",
                        )
                    text = delta.get("content", "")
                    reason = choice.get("finish_reason")
                    if finish is not None and (text or reason is not None):
                        raise ServiceError(
                            "Hikari sent content after finishing the reply.",
                            502,
                            "invalid_stream",
                        )
                    if text:
                        length += len(text)
                        if length > MAX_REPLY_CHARACTERS:
                            raise ServiceError(
                                "The reply exceeded Kadan's one-million-character limit.",
                                413,
                                "reply_too_large",
                            )
                        content.append(text)
                        yield event("delta", {"content": text})
                    if reason is not None:
                        if reason not in {"stop", "length"}:
                            raise ServiceError(
                                "Hikari reported an unsuccessful finish.",
                                502,
                                "invalid_stream",
                            )
                        finish = reason
            if not done or finish is None:
                raise ServiceError(
                    "Hikari disconnected before completing the reply. The partial reply was not saved.",
                    502,
                    "incomplete_stream",
                )
            if not content:
                raise ServiceError(
                    "Hikari completed without a reply. No assistant message was saved.",
                    502,
                    "empty_reply",
                )
            assistant = Message(role="assistant", content="".join(content))
            stored = await self.append(key, saved.id, saved.version, assistant)
            yield event(
                "done",
                {
                    "conversation": stored.model_dump(mode="json"),
                    "finish_reason": finish,
                },
            )
        except httpx.RequestError:
            yield event(
                "error",
                {
                    "code": "inference_unavailable",
                    "message": "Hikari disconnected or timed out. Your message is saved; the partial reply was not saved.",
                    "status": 503,
                    "user_saved": True,
                },
            )
        except ServiceError as error:
            message = str(error)
            if error.code == "version_conflict":
                message = "The conversation changed during this turn. Saved messages and completed tool actions remain; this reply was not saved. Reload the conversation."
            yield event(
                "error",
                {
                    "code": error.code,
                    "message": message,
                    "status": error.status,
                    "user_saved": True,
                },
            )


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


def completion_messages(
    conversation: Resource, persona: Resource | None, user: Message
) -> list[CompletionMessage]:
    messages = []
    if persona and persona.markdown.strip():
        messages.append(CompletionMessage(role="system", content=persona.markdown))
    for message in [*conversation.messages, user]:
        if message.role == "tool":
            messages.append(tool_history(message.content))
            continue
        if not message.content.strip():
            raise ServiceError(
                "This conversation contains an empty message that Hikari cannot replay. Start a new conversation.",
                422,
                "unsupported_history",
            )
        messages.append(CompletionMessage(role=message.role, content=message.content))
    if (
        len(messages) > MAX_PROMPT_MESSAGES
        or sum(len(message.content) for message in messages) > MAX_PROMPT_CHARACTERS
    ):
        raise ServiceError(
            "This conversation exceeds Kadan's 100-message or 48,000-character chat limit. Start a new conversation. Saved history is unchanged.",
            422,
            "history_too_large",
        )
    return messages


def event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def sse_data(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    data: list[str] = []
    size = 0
    async for line in lines:
        if line == "":
            if data:
                yield "\n".join(data)
            data = []
            size = 0
        elif line.startswith("data:"):
            value = line[5:].removeprefix(" ")
            size += len(value)
            if size > MAX_REPLY_CHARACTERS:
                raise ServiceError(
                    "Hikari returned an oversized chat event.", 502, "invalid_stream"
                )
            data.append(value)
    # A frame without its terminating blank line is not a complete SSE event.
