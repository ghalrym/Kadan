"""Conversation work resumes on durable user or image events, never idle inference."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING
from uuid import UUID, uuid4, uuid5

from kadan_tracing import traced, traced_job
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from langgraph.errors import NodeCancelledError
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from ..api.config import ServiceError
from ..api.group_chat_mcp import GroupChatMcpHandler
from ..api.memory_context import (
    build_core_memories_snapshot,
    extract_recall_terms,
    fetch_active_memories,
    format_core_memories_section,
    persist_memory_state,
    recall_archival_memories,
    select_core_memories,
)
from ..api.pydantic_models.agent_runs import (
    AgentRun,
    AgentRunState,
    ImageJob,
    RunCall,
    RunMessage,
    RunObservation,
    RunResult,
    TranscriptEntry,
    UnrecoveredRecord,
)
from ..api.pydantic_models.mcp import McpServerRecord
from ..api.pydantic_models.resources import Message, Resource, ResourceKind
from ..api.run_client import RunClient
from ..api.services import character_persona_section, parse_resource
from .agent_inference import (
    REASONING_TOKENS,
    REPLY_TOKENS,
    AgentInference,
    ToolCommand,
    tool_characters,
)
from .compaction import (
    IN_RUN_RECENT_TAIL_MESSAGES,
    format_summary_section,
    generate_rolling_summary,
    persist_in_run_compaction,
    should_compact_in_run,
    split_uncompacted,
)
from .conversation_history import (
    ReadHistoryArguments,
    SearchHistoryArguments,
    history_page,
)
from .mcp_runtime import (
    McpOrigin,
    build_mcp_tools,
    execute_mcp_tool,
    find_server_tool,
    search_mcp_tools,
    validate_json_schema_arguments,
)
from .memories import AgentMemories
from .tools import (
    ALL_TOOLS,
    CONVERSATION_TOOLS,
    GROUP_CHAT_TOOLS,
    GROUP_CHAT_TOOLS_BY_NAME,
    MEMORY_TOOLS,
    PERSONA_TOOLS,
    TOOLS_BY_NAME,
)
from .tools.conversation import GenerateImageArguments, SendMessageArguments
from .tools.group_chat import (
    GroupChatReadMessagesArguments,
    GroupChatSendMessageArguments,
)
from .tools.mcp import MCP_GATEWAY_TOOLS, McpCallArguments, McpSearchArguments
from .tools.persona import RenameSelfArguments, UpdatePersonaArguments
from .tools.updates import InstallAppUpdateArguments
from .tools.workspace import (
    CreateWorkspaceDocumentArguments,
    ListWorkspaceFilesArguments,
    ReadWorkspaceFileArguments,
)
from .transcript import entry_characters, transcript_messages
from .workspace_files import WorkspaceFiles

if TYPE_CHECKING:
    from ..api.services import LocalServices

logger = logging.getLogger(__name__)
LEASE_SECONDS = 60
MAX_COMMANDS = 24
MAX_ARGUMENT_CORRECTIONS = 2
MAX_ERROR_RETRIES = 3
FALLBACK_FAILURE_REPLY = (
    "I ran into trouble completing that request. "
    "Let me know if you'd like me to try again, or ask if you want more details about what happened."
)
# Transcript budget in characters when Hikari does not report a context window.
MAX_CONTEXT = 48000
# Tool results are JSON, which tokenizes densely; three characters per token
# estimates transcript space. Reserve generation separately from template overhead.
CHARACTERS_PER_TOKEN = 3
RESERVED_TOKENS = REASONING_TOKENS + REPLY_TOKENS + 1024


def is_lease_lost(error: BaseException) -> bool:
    return (
        isinstance(error, ServiceError)
        and error.status == 409
        and error.code == "lease_lost"
    )


def is_group_chat_run(run: AgentRun) -> bool:
    return run.trigger_event is not None and run.trigger_event.kind == "group_chat"


def context_characters(context_limit: int | None) -> int:
    """The agent's transcript budget, sized to the selected model's window."""
    if context_limit is None:
        return MAX_CONTEXT
    return max(0, (context_limit - RESERVED_TOKENS) * CHARACTERS_PER_TOKEN)


class ArgumentProblem(BaseModel):
    field: str
    code: str


class ArgumentRejection(BaseModel):
    tool: str
    problems: list[ArgumentProblem]
    error: str = (
        "No tool was executed. Correct the arguments using the bound tool schema; "
        "do not repeat the unchanged call."
    )


def validate(
    command: ToolCommand, tools: list[StructuredTool]
) -> ArgumentRejection | None:
    """Reject before execution, retaining field diagnostics without input values."""
    tools_by_name = {tool.name: tool for tool in tools}
    if command.tool not in tools_by_name:
        raise ServiceError(f"unknown tool {command.tool!r}", 502, "invalid_command")
    schema = tools_by_name[command.tool].args_schema
    if isinstance(schema, dict):
        problems = validate_json_schema_arguments(schema, command.arguments)
        if problems:
            return ArgumentRejection(
                tool=command.tool,
                problems=[
                    ArgumentProblem(
                        field=field
                        if re.fullmatch(
                            r"[A-Za-z_][A-Za-z0-9_]{0,63}|[0-9]{1,6}", field
                        )
                        else "[redacted]",
                        code=code,
                    )
                    for field, code in problems
                ],
            )
        return None
    try:
        schema.model_validate(command.arguments)
    except ValidationError as error:
        return ArgumentRejection(
            tool=command.tool,
            problems=[
                ArgumentProblem(
                    field=".".join(
                        str(part)
                        if re.fullmatch(
                            r"[A-Za-z_][A-Za-z0-9_]{0,63}|[0-9]{1,6}", str(part)
                        )
                        else "[redacted]"
                        for part in problem["loc"][:4]
                    )
                    or "arguments",
                    code=problem["type"],
                )
                for problem in error.errors(
                    include_url=False, include_input=False, include_context=False
                )[:5]
            ],
        )
    return None


class RunConversation:
    """The conversation tools' view of one run, at the step being executed."""

    def __init__(self, runs: AgentRuns, run: AgentRun, replayed: bool = False):
        self.runs = runs
        self.run = run
        self.action_id = uuid5(run.id, f"command:{len(run.transcript)}")
        self.end_run = False
        self.finished = False
        self.replayed = replayed
        self.post_save_action: Callable[[], Awaitable[None]] | None = None

    async def check_app_update(self) -> dict:
        updates = getattr(self.runs.services, "updates", None)
        if updates is None:
            return {
                "tool": "check_app_update",
                "error": "Update checks are not available in this environment.",
            }
        try:
            result = await updates.check()
        except ServiceError as error:
            return {"tool": "check_app_update", "error": str(error)}
        return {"tool": "check_app_update", **result.model_dump(mode="json")}

    async def install_app_update(self, args: InstallAppUpdateArguments) -> dict:
        if self.replayed:
            return {
                "tool": "install_app_update",
                "updated": False,
                "replayed": True,
                "error": (
                    "Kadan restarted before this update finished reporting back. "
                    "Call check_app_update to see the currently installed version "
                    "before trying again."
                ),
            }
        updates = getattr(self.runs.services, "updates", None)
        if updates is None:
            return {
                "tool": "install_app_update",
                "updated": False,
                "error": "Automatic updates are not available in this environment.",
            }
        try:
            staged = await updates.stage(target_version=args.target_version)
        except ServiceError as error:
            return {
                "tool": "install_app_update",
                "updated": False,
                "error": str(error),
            }

        message = Message(
            id=self.action_id, role="assistant", content=args.announcement
        )
        async with self.runs.lock:
            await self.runs.services.json_request(
                "Tsuchi",
                "POST",
                self.runs.path(self.run.agent_key, self.run.id) + "/messages",
                json={
                    "worker_id": str(self.runs.owner),
                    "message": message.model_dump(mode="json"),
                    "image_job_id": None,
                },
            )
        self.end_run = True

        async def _apply_after_save() -> None:
            await updates.apply(
                target_version=staged.target_version, include_desktop=True
            )

        self.post_save_action = _apply_after_save
        return {
            "tool": "install_app_update",
            "updated": True,
            "target_version": staged.target_version,
            "server_restarting": staged.server_staged,
            "desktop_update_requested": staged.desktop_update_available,
            "delivered": True,
            "message": staged.message,
        }

    async def search_mcp_tools(self, args: McpSearchArguments) -> dict:
        servers = await self.runs.active_mcp_servers()
        return search_mcp_tools(servers, server_filter=args.server, query=args.query)

    async def call_mcp_tool(self, args: McpCallArguments) -> dict:
        servers = await self.runs.active_mcp_servers()
        match = find_server_tool(servers, args.server, args.tool)
        if match is None:
            return {
                "error": f"Tool '{args.tool}' was not found on connected MCP server '{args.server}'. "
                "Use mcp_search_tools to check available servers and tools."
            }
        server, tool_info = match
        return await execute_mcp_tool(
            self.runs.services,
            server,
            tool_info,
            args.arguments,
            action_id=self.action_id,
            replayed=self.replayed,
            origin=McpOrigin(
                agent_key=self.run.agent_key,
                conversation_id=self.run.conversation_id,
                run_id=self.run.id,
            ),
        )

    async def send_message(self, args: SendMessageArguments) -> dict:
        if is_group_chat_run(self.run):
            return {
                "tool": "send_message",
                "delivered": False,
                "error": (
                    "Group Chat runs cannot send messages to 1-on-1 Chat. "
                    "Use group_chat_send_message to reply in Group Chat, or call finish."
                ),
            }
        message = Message(id=self.action_id, role="assistant", content=args.content)
        async with self.runs.lock:
            await self.runs.services.json_request(
                "Tsuchi",
                "POST",
                self.runs.path(self.run.agent_key, self.run.id) + "/messages",
                json={
                    "worker_id": str(self.runs.owner),
                    "message": message.model_dump(mode="json"),
                    "image_job_id": str(args.image_job_id)
                    if args.image_job_id
                    else None,
                },
            )
        self.end_run = args.end_run
        return {"tool": "send_message", "delivered": True}

    async def _group_chat_participant(self):
        group_chat = getattr(self.runs.services, "group_chat", None)
        if group_chat is None:
            raise ServiceError(
                "Group Chat is unavailable.", 503, "group_chat_unavailable"
            )
        room_id: UUID | None = None
        if self.run.trigger_event and self.run.trigger_event.kind == "group_chat":
            try:
                parsed = json.loads(self.run.trigger_event.content)
                if isinstance(parsed, dict) and parsed.get("room_id"):
                    room_id = UUID(str(parsed["room_id"]))
            except (ValueError, TypeError):
                room_id = None
        if room_id is None:
            room = await group_chat.default_room()
            room_id = room.id
        participant = await group_chat.ensure_kadan_participant(
            room_id, agent_key=self.run.agent_key
        )
        if participant is None:
            raise ServiceError(
                "Could not resolve Kadan's Group Chat participant identity.",
                502,
                "participant_missing",
            )
        return participant

    async def group_chat_list_participants(self) -> dict:
        try:
            participant = await self._group_chat_participant()
            handler = GroupChatMcpHandler(self.runs.services)
            result, _ = await handler.call_tool(
                participant, "group_chat_list_participants", {}
            )
            return result
        except ServiceError as error:
            if is_lease_lost(error):
                raise
            return {"tool": "group_chat_list_participants", "error": str(error)}

    async def group_chat_read_messages(
        self, args: GroupChatReadMessagesArguments
    ) -> dict:
        try:
            participant = await self._group_chat_participant()
            handler = GroupChatMcpHandler(self.runs.services)
            result, _ = await handler.call_tool(
                participant,
                "group_chat_read_messages",
                args.model_dump(mode="json", exclude_none=True),
            )
            return result
        except ServiceError as error:
            if is_lease_lost(error):
                raise
            return {"tool": "group_chat_read_messages", "error": str(error)}

    async def group_chat_send_message(
        self, args: GroupChatSendMessageArguments
    ) -> dict:
        await self.runs.owned(self.run)
        try:
            participant = await self._group_chat_participant()
            handler = GroupChatMcpHandler(self.runs.services)
            payload = args.model_dump(mode="json", exclude_none=True)
            # Deterministic idempotency key from saved action_id prevents duplicate posts on restart.
            payload["client_request_id"] = str(self.action_id)
            result, is_error = await handler.call_tool(
                participant, "group_chat_send_message", payload
            )
            if not is_error:
                self.end_run = True
            return result
        except ServiceError as error:
            if is_lease_lost(error):
                raise
            return {"tool": "group_chat_send_message", "error": str(error)}

    async def generate_image(self, args: GenerateImageArguments) -> dict:
        image_status = await self.runs.services.image_inference_status()
        if image_status.state != "ready":
            return {
                "tool": "generate_image",
                "available": False,
                "status": image_status.state,
                "download_state": image_status.download_state,
                "user_instructions": image_status.message,
            }
        image = ImageJob.model_validate(
            await self.runs.services.json_request(
                "Tsuchi",
                "POST",
                self.runs.path(self.run.agent_key, self.run.id) + "/image-jobs",
                json={
                    "id": str(self.action_id),
                    "prompt": args.prompt,
                    "request_id": str(uuid5(self.action_id, "request")),
                    "worker_id": str(self.runs.owner),
                },
            )
        )
        self.runs.services.images.wakeup.set()
        self.end_run = True
        return {
            "tool": "generate_image",
            "job_id": str(image.id),
            "status": image.status,
            "delivered": image.delivered_message_id is not None,
        }

    async def conversation_history(
        self, args: ReadHistoryArguments | SearchHistoryArguments
    ) -> dict:
        try:
            async with self.runs.lock:
                current = await self.runs.owned(self.run)
                if (current.agent_key, current.conversation_id) != (
                    self.run.agent_key,
                    self.run.conversation_id,
                ):
                    raise ValueError("Run conversation scope changed.")
                resource = await self.runs.services.resource(
                    self.run.agent_key, self.run.conversation_id
                )
                if (
                    resource.id != self.run.conversation_id
                    or resource.agent_key != self.run.agent_key
                    or resource.kind != ResourceKind.CONVERSATION
                    or resource.deleted_at is not None
                ):
                    raise ValueError("The run conversation is no longer available.")
                return history_page(resource, args)
        except (ServiceError, ValueError) as error:
            return {"error": str(error), "history_available": False}

    async def read_task_status(self, offset: int) -> dict:
        result = await self.runs.services.json_request(
            "Tsuchi",
            "GET",
            f"/api/agents/{self.run.agent_key}/resources/{self.run.conversation_id}/task-status",
            params={"offset": offset},
        )
        image_status = await self.runs.services.image_inference_status()
        if image_status.state != "ready" and isinstance(result, dict):
            return {
                **result,
                "image_generation": {
                    "available": False,
                    "status": image_status.state,
                    "download_state": image_status.download_state,
                    "user_instructions": image_status.message,
                },
            }
        return result

    async def finish(self) -> None:
        self.finished = True

    async def read_persona(self) -> dict:
        persona = await self.runs.services.agent_persona(self.run.agent_key)
        return {
            "persona": persona.model_dump(
                mode="json", include={"id", "title", "markdown", "version"}
            )
            if persona
            else None
        }

    async def update_persona(self, args: UpdatePersonaArguments) -> dict:
        # Stop or loss of the lease must fence a write that has not started yet.
        await self.runs.owned(self.run)
        expected_version = args.expected_version
        if expected_version is None:
            current = await self.runs.services.agent_persona(self.run.agent_key)
            if current is not None:
                expected_version = current.version
            await self.runs.owned(self.run)
        persona = await self.runs.services.save_agent_persona(
            self.run.agent_key,
            markdown=args.markdown,
            expected_version=expected_version,
            title=args.title,
        )
        return {
            "persona": persona.model_dump(
                mode="json", include={"id", "title", "markdown", "version"}
            )
        }

    async def rename_self(self, args: RenameSelfArguments) -> dict:
        # Stop or loss of the lease must fence a write that has not started yet.
        await self.runs.owned(self.run)
        current = await self.runs.services.agent(self.run.agent_key)
        if current.name == args.name:
            return {
                "agent_key": current.key,
                "name": current.name,
                "changed": False,
            }
        await self.runs.owned(self.run)
        updated = await self.runs.services.rename_agent(
            self.run.agent_key,
            name=args.name,
            expected_name=current.name,
        )
        return {
            "agent_key": updated.key,
            "name": updated.name,
            "changed": True,
        }

    async def read_workspace_file(self, args: ReadWorkspaceFileArguments) -> dict:
        await self.runs.owned(self.run)
        try:
            saved = await WorkspaceFiles(self.runs.services, self.run.agent_key).read(
                args.file_id
            )
        except ServiceError as error:
            return {
                "tool": "read_workspace_file",
                "file_id": str(args.file_id),
                "error": str(error),
            }
        return {"tool": "read_workspace_file", **saved}

    async def list_workspace_files(self, args: ListWorkspaceFilesArguments) -> dict:
        result = {"tool": "list_workspace_files"}
        await self.runs.owned(self.run)
        try:
            found = await WorkspaceFiles(self.runs.services, self.run.agent_key).list(
                query=args.query, limit=args.limit, offset=args.offset
            )
        except ServiceError as error:
            return {**result, "error": str(error)}
        return {**result, **found}

    async def create_workspace_document(
        self, args: CreateWorkspaceDocumentArguments
    ) -> dict:
        result = {"tool": "create_workspace_document"}
        # Resource creation has no idempotency key. A saved command without a
        # result may already have created its document before a runner restart.
        if self.replayed:
            return {
                **result,
                "error": "The runner restarted before document creation reported "
                "a result. It may have been saved; check Workspace before retrying.",
            }
        async with self.runs.lock:
            await self.runs.owned(self.run)
            try:
                document = parse_resource(
                    await self.runs.services.json_request(
                        "Tsuchi",
                        "POST",
                        f"/api/agents/{self.run.agent_key}/resources",
                        json={"kind": "space", **args.model_dump(mode="json")},
                    )
                )
            except ServiceError as error:
                message = str(error)
                if error.status >= 500:
                    message += " The document may have been saved; check Workspace before retrying."
                return {**result, "error": message}
        return {
            **result,
            "document_id": str(document.id),
            "title": document.title,
            "parent_id": str(document.parent_id) if document.parent_id else None,
            "version": document.version,
        }


class RunState(BaseModel):
    run: AgentRun
    command: ToolCommand | None = None
    # True when the command was read back from the transcript after a restart.
    replayed: bool = False


class AgentRuns(RunClient):
    def __init__(self, services: LocalServices, owner: UUID | None = None):
        super().__init__(services)
        self.owner = owner or uuid4()
        self.wakeup = asyncio.Event()
        self.lock = asyncio.Lock()
        self.inference = AgentInference(services.client, services.settings.hikari_url)
        graph = StateGraph(RunState)
        graph.add_node("prepare", self.prepare)
        graph.add_node("model", self.infer_command)
        graph.add_node("tool", self.invoke_tool)
        graph.add_edge(START, "prepare")
        graph.add_edge("prepare", "model")
        graph.add_conditional_edges("model", self.after_model)
        graph.add_conditional_edges("tool", self.after_tool)
        # Tsuchi owns checkpoints and side-effect receipts. A second persistence
        # store here would compete with its version/lease fencing.
        self.graph = graph.compile()

    async def save(self, run: AgentRun, **changes) -> AgentRun:
        state = AgentRunState.model_validate({**run.model_dump(), **changes})
        value = await self.services.json_request(
            "Tsuchi",
            "PUT",
            self.path(run.agent_key, run.id),
            json={
                **state.model_dump(mode="json"),
                "transcript_format": 2,
                "expected_version": run.version,
                "worker_id": str(self.owner),
            },
        )
        return AgentRun.model_validate(value)

    async def owned(self, run: AgentRun) -> AgentRun:
        current = await self.get(run.agent_key, run.id)
        if (
            (current.id, current.agent_key, current.conversation_id)
            != (run.id, run.agent_key, run.conversation_id)
            or current.status != "running"
            or current.owner_id != self.owner
            or current.lease_expires_at is None
            or current.lease_expires_at <= datetime.now(UTC)
        ):
            raise ServiceError(
                "This agent run is no longer owned by this worker.", 409, "lease_lost"
            )
        return current

    async def update(self, run: AgentRun, **changes) -> AgentRun:
        async with self.lock:
            return await self.save(await self.owned(run), **changes)

    async def renew_lease(self, run: AgentRun) -> None:
        while True:
            await asyncio.sleep(10)
            await self.update(
                run,
                lease_expires_at=datetime.now(UTC) + timedelta(seconds=LEASE_SECONDS),
            )

    @traced_job("kadan.agent_runs.process")
    async def process(self, run: AgentRun) -> None:
        try:
            run = await self.save(
                run,
                status="running",
                owner_id=self.owner,
                lease_expires_at=datetime.now(UTC) + timedelta(seconds=LEASE_SECONDS),
                error=None,
            )
        except ServiceError as error:
            if error.status == 409:
                return
            raise
        self._preempt_memory_queue()
        now = datetime.now(UTC)
        queue_wait = max(0.0, (now - run.created_at).total_seconds())
        logger.info(
            "Agent run claimed run=%s at=%s queue_wait_seconds=%.3f",
            run.id,
            now.isoformat(),
            queue_wait,
        )
        work = asyncio.create_task(self.advance(run))
        heartbeat = asyncio.create_task(self.renew_lease(run))
        try:
            done, _ = await asyncio.wait(
                {work, heartbeat}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()
        except (ServiceError, ValidationError, OSError) as error:
            if is_lease_lost(error):
                logger.info(
                    "Run ended after cancellation or loss of ownership: %s", run.id
                )
            else:
                await self.handle_failure(run, error)
        finally:
            work.cancel()
            heartbeat.cancel()
            await asyncio.gather(work, heartbeat, return_exceptions=True)

    async def handle_failure(self, run: AgentRun, error: BaseException) -> None:
        error_detail = str(error)[:2000]
        try:
            current = await self.owned(run)
        except ServiceError:
            logger.info("Run ended after cancellation or loss of ownership: %s", run.id)
            return
        if is_group_chat_run(current):
            # Never leak Group Chat failure notices into the user's private 1-on-1 Chat.
            try:
                await self.update(
                    current,
                    status="failed",
                    owner_id=None,
                    lease_expires_at=None,
                    error=error_detail,
                )
            except ServiceError:
                logger.info(
                    "Run ended after cancellation or loss of ownership: %s",
                    current.id,
                )
            return
        reply_id = uuid5(current.id, "failure_reply")
        if reply_id not in current.delivered_message_ids:
            reply_text = await self._compose_failure_reply(current, error_detail)
            try:
                await self._deliver_failure_reply(current, reply_id, reply_text)
            except ServiceError:
                logger.info(
                    "Run could not deliver failure reply: %s",
                    current.id,
                )
                try:
                    await self.update(
                        current,
                        status="failed",
                        owner_id=None,
                        lease_expires_at=None,
                        error=error_detail,
                    )
                except ServiceError:
                    logger.info(
                        "Run ended after cancellation or loss of ownership: %s",
                        current.id,
                    )
                return
        try:
            await self.update(
                current,
                status="completed",
                owner_id=None,
                lease_expires_at=None,
                error=error_detail,
            )
        except ServiceError:
            logger.info(
                "Run ended after cancellation or loss of ownership: %s", current.id
            )

    async def _deliver_failure_reply(
        self, run: AgentRun, message_id: UUID, content: str
    ) -> None:
        message = Message(id=message_id, role="assistant", content=content)
        async with self.lock:
            await self.services.json_request(
                "Tsuchi",
                "POST",
                self.path(run.agent_key, run.id) + "/messages",
                json={
                    "worker_id": str(self.owner),
                    "message": message.model_dump(mode="json"),
                    "image_job_id": None,
                },
            )

    @staticmethod
    def _failure_context_messages(error_detail: str) -> tuple[list[BaseMessage], int]:
        call_id = "run_failure_context"
        payload = {
            "status": "failed_after_retries",
            "error": error_detail,
            "instruction": (
                "You encountered an error and could not complete the user's request after retrying. "
                "Reply briefly and naturally to let the user know you struggled to complete the task "
                "and offer to try again or help another way. "
                "Do not tell the user the exact technical error or internal details unless they ask for details."
            ),
        }
        serialized = json.dumps(payload, ensure_ascii=False)
        return (
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "run_failure_context", "args": {}, "id": call_id}
                    ],
                ),
                ToolMessage(
                    content=serialized,
                    name="run_failure_context",
                    tool_call_id=call_id,
                ),
            ],
            len(serialized) + 96,
        )

    async def _compose_failure_reply(self, run: AgentRun, error_detail: str) -> str:
        if not run.transcript or any(
            isinstance(item, UnrecoveredRecord) for item in run.transcript
        ):
            return FALLBACK_FAILURE_REPLY
        try:
            status = await self.services.inference_status()
            if status.state != "ready":
                return FALLBACK_FAILURE_REPLY
            send_only = [TOOLS_BY_NAME["send_message"]]
            limit = context_characters(
                getattr(status, "context_limit", None)
            ) - tool_characters(send_only)
            if limit <= 0:
                return FALLBACK_FAILURE_REPLY
            safe_transcript = list(run.transcript)
            if safe_transcript and isinstance(safe_transcript[-1], RunCall):
                safe_transcript.pop()
            failure_obs, failure_chars = self._failure_context_messages(error_detail)
            transcript_chars = sum(entry_characters(item) for item in safe_transcript)
            if (
                "context" not in error_detail.lower()
                and transcript_chars + failure_chars <= limit
            ):
                messages = [*transcript_messages(safe_transcript), *failure_obs]
            else:
                system_text = (
                    "You are the user's assistant. You struggled to complete the previous request. "
                    "Reply briefly and naturally to let the user know you struggled to complete the task "
                    "without sharing the exact technical error unless they ask for details."
                )
                last_user = next(
                    (
                        item.content
                        for item in reversed(safe_transcript)
                        if isinstance(item, RunMessage) and item.role == "user"
                    ),
                    None,
                )
                minimal_chars = (
                    len(system_text)
                    + (min(len(last_user), 1000) if last_user else 0)
                    + failure_chars
                )
                if minimal_chars > limit:
                    return FALLBACK_FAILURE_REPLY
                messages = [SystemMessage(content=system_text)]
                if last_user:
                    messages.append(HumanMessage(content=last_user[:1000]))
                messages.extend(failure_obs)
            _, command, rejection = await self._infer_once(
                run, messages, status.model_id, send_only
            )
            if rejection is None and command.tool == "send_message":
                content = command.arguments.get("content", "")
                if isinstance(content, str) and content.strip():
                    return content
        except (ServiceError, ValidationError, OSError) as reply_error:
            logger.warning(
                "Failure reply inference failed for run=%s: %s", run.id, reply_error
            )
        return FALLBACK_FAILURE_REPLY

    async def recent_run_errors(
        self,
        run: AgentRun,
        messages: list[Message],
        included_message_ids: set[UUID],
    ) -> list[dict]:
        try:
            values = await self.services.json_request(
                "Tsuchi",
                "GET",
                f"/api/agents/{run.agent_key}/resources/{run.conversation_id}/runs",
            )
        except ServiceError:
            return []
        if not isinstance(values, list):
            return []
        prior_runs: list[AgentRun] = []
        for value in values:
            try:
                item = AgentRun.model_validate(value)
            except ValidationError:
                continue
            if (
                item.id == run.id
                or item.agent_key != run.agent_key
                or item.conversation_id != run.conversation_id
                or not item.error
                or item.created_at > run.created_at
            ):
                continue
            if (
                item.trigger_message_id is not None
                and item.trigger_message_id not in included_message_ids
            ):
                continue
            prior_runs.append(item)
        prior_runs.sort(key=lambda item: (item.created_at, item.id))
        messages_by_id = {message.id: message for message in messages}
        records: list[dict] = []
        for item in prior_runs[-3:]:
            trigger = (
                messages_by_id.get(item.trigger_message_id)
                if item.trigger_message_id is not None
                else None
            )
            record: dict[str, object] = {
                "occurred_at": item.updated_at.isoformat(),
                "error": item.error,
            }
            if trigger is not None:
                record["request_excerpt"] = trigger.content.strip()[:160]
            records.append(record)
        return records

    async def _compact_in_run(
        self,
        run: AgentRun,
        conversation: Resource,
        messages: list[Message],
        uncompacted: list[Message],
        compacted_summary: str | None,
        inference_status,
    ) -> tuple[Resource, str | None, list[Message]]:
        to_compact = [
            message
            for message in uncompacted[:-IN_RUN_RECENT_TAIL_MESSAGES]
            if message.id != run.trigger_message_id
        ]
        if not to_compact:
            return conversation, compacted_summary, uncompacted
        try:
            status = inference_status or await self.services.inference_status()
            if status.state != "ready":
                return conversation, compacted_summary, uncompacted
            new_summary = await generate_rolling_summary(
                self.inference,
                agent_key=run.agent_key,
                conversation_id=run.conversation_id,
                existing_summary=compacted_summary,
                messages=to_compact,
                model_id=status.model_id,
                context_limit=getattr(status, "context_limit", None),
                priority="interactive",
            )
            snapshot_kwargs: dict[str, object] = {}
            if hasattr(self.services, "list_resource_kinds"):
                fresh_snapshot, _ = await build_core_memories_snapshot(
                    self.services, run.agent_key
                )
                snapshot_kwargs = {
                    "core_memories_snapshot": fresh_snapshot,
                    "core_memories_refreshed_at": datetime.now(UTC),
                }
            updated = await persist_in_run_compaction(
                self.services,
                agent_key=run.agent_key,
                conversation=conversation,
                compacted_summary=new_summary,
                compacted_through_message_id=to_compact[-1].id,
                verify_owned=(lambda: self.owned(run))
                if run.status == "running"
                else None,
                **snapshot_kwargs,
            )
            updated_messages = updated.messages
            if run.trigger_message_id is not None:
                target = next(
                    (
                        index
                        for index, item in enumerate(updated_messages)
                        if item.id == run.trigger_message_id and item.role == "user"
                    ),
                    None,
                )
                if target is not None:
                    updated_messages = updated_messages[: target + 1]
            summary, remaining = split_uncompacted(
                updated_messages,
                updated.compacted_summary,
                updated.compacted_through_message_id,
            )
            if run.trigger_message_id is not None and not any(
                item.id == run.trigger_message_id for item in remaining
            ):
                remaining = [messages[-1]]
            logger.info(
                "In-run compacted conversation agent=%s conversation=%s through=%s compacted=%d remaining=%d",
                run.agent_key,
                run.conversation_id,
                to_compact[-1].id,
                len(to_compact),
                len(remaining),
            )
            return updated, summary, remaining
        except (ServiceError, ValidationError, OSError) as error:
            if is_lease_lost(error):
                raise
            logger.warning(
                "In-run compaction failed for run=%s; continuing with tail history: %s",
                run.id,
                error,
            )
            return conversation, compacted_summary, uncompacted

    @traced("kadan.agent_runs.context")
    async def context(
        self,
        run: AgentRun,
        limit: int | None = None,
        *,
        inference_status=None,
    ) -> list[TranscriptEntry]:
        instruction_file = Path(self.services.settings.agent_instruction_file)
        try:
            instructions = await asyncio.to_thread(instruction_file.read_text)
        except UnicodeError as error:
            raise ServiceError(
                f"The shared agent instructions file {instruction_file} could not "
                "be decoded. Check its text encoding."
            ) from error
        conversation = await self.services.resource(run.agent_key, run.conversation_id)
        if conversation.kind != ResourceKind.CONVERSATION or conversation.deleted_at:
            raise ServiceError("The conversation is no longer available.", 409)
        agent = await self.services.agent(run.agent_key)
        owned_persona = await self.services.agent_persona(run.agent_key)
        instructions += "\n\n" + character_persona_section(
            agent.name, owned_persona.markdown if owned_persona else None
        )
        base_system = RunMessage(role="system", content=instructions)
        messages = conversation.messages
        # A later queued request may already be saved when this one starts.
        if run.trigger_message_id is not None:
            target = next(
                (
                    i
                    for i, item in enumerate(messages)
                    if item.id == run.trigger_message_id and item.role == "user"
                ),
                None,
            )
            if target is None:
                raise ServiceError(
                    "The message for this response is no longer available.", 422
                )
            messages = messages[: target + 1]

        event = None
        if run.trigger_job_id:
            image = ImageJob.model_validate(
                await self.services.json_request(
                    "Tsuchi",
                    "GET",
                    f"/api/agents/{run.agent_key}/image-jobs/{run.trigger_job_id}",
                )
            )
            if (image.agent_key, image.conversation_id) != (
                run.agent_key,
                run.conversation_id,
            ):
                raise ServiceError(
                    "The image belongs to a different conversation.", 409
                )
            event = RunObservation(
                name="image_job_event",
                data={
                    "job_id": str(image.id),
                    "status": image.status,
                    "prompt": image.prompt,
                    "error": image.error,
                    "delivered": image.delivered_message_id is not None,
                },
            )
        elif run.trigger_event:
            try:
                detail = json.loads(run.trigger_event.content)
            except ValueError:
                detail = run.trigger_event.content
            if run.trigger_event.kind == "group_chat":
                event_data = await self._build_group_chat_event_data(run, detail)
                event = RunObservation(
                    name="group_chat_event",
                    data=event_data,
                )
            else:
                event = RunObservation(
                    name={
                        # Legacy: computer results queued before its removal.
                        "computer": "computer_task_event",
                        "codex": "codex_thread_event",
                        "t3": "t3_thread_event",
                        "work": "work_thread_event",
                    }[run.trigger_event.kind],
                    data=detail if isinstance(detail, dict) else {"detail": detail},
                )

        if limit is None:
            mcp_servers = await self.active_mcp_servers()
            limit = MAX_CONTEXT - tool_characters(self.bound_tools(run, mcp_servers))
        base_budget = (
            limit
            - entry_characters(base_system)
            - (entry_characters(event) if event else 0)
        )
        if base_budget < 0:
            raise ServiceError(
                "Agent instructions and event exceed the context limit.", 422
            )

        compacted_summary, uncompacted = split_uncompacted(
            messages,
            conversation.compacted_summary,
            conversation.compacted_through_message_id,
        )
        if run.trigger_message_id is not None and not any(
            item.id == run.trigger_message_id for item in uncompacted
        ):
            uncompacted = [messages[-1]]

        uncompacted_chars = sum(
            entry_characters(
                RunMessage(
                    role=item.role,
                    content=item.content,
                    attachment_ids=item.attachment_ids
                    if item.id == run.trigger_message_id
                    else [],
                )
            )
            for item in uncompacted
        )
        effective_core_snapshot = conversation.core_memories_snapshot
        core_ids: set[UUID] | None = None
        will_compact = should_compact_in_run(
            uncompacted,
            uncompacted_chars=uncompacted_chars,
            summary_chars=len(format_summary_section(compacted_summary))
            + len(format_core_memories_section(effective_core_snapshot)),
            available_budget=base_budget,
        )
        if will_compact:
            conversation, compacted_summary, uncompacted = await self._compact_in_run(
                run,
                conversation,
                messages,
                uncompacted,
                compacted_summary,
                inference_status,
            )
            effective_core_snapshot = conversation.core_memories_snapshot

        if effective_core_snapshot is None and hasattr(
            self.services, "list_resource_kinds"
        ):
            built_snapshot, core_ids = await build_core_memories_snapshot(
                self.services, run.agent_key
            )
            effective_core_snapshot = built_snapshot
            try:
                persisted = await persist_memory_state(
                    self.services,
                    agent_key=run.agent_key,
                    conversation_id=conversation.id,
                    core_memories_snapshot=built_snapshot,
                    core_memories_refreshed_at=datetime.now(UTC),
                    only_if_snapshot_unset=True,
                    set_snapshot=True,
                )
                if persisted is not None:
                    conversation = persisted
                    if conversation.core_memories_snapshot is not None:
                        effective_core_snapshot = conversation.core_memories_snapshot
            except ServiceError:
                logger.debug(
                    "Could not persist initial core memories snapshot for conversation=%s",
                    conversation.id,
                )

        trigger_msg = (
            next(
                (
                    item
                    for item in reversed(uncompacted)
                    if item.id == run.trigger_message_id
                ),
                None,
            )
            if run.trigger_message_id is not None
            else None
        )
        trigger_size = (
            entry_characters(
                RunMessage(
                    role=trigger_msg.role,
                    content=trigger_msg.content,
                    attachment_ids=trigger_msg.attachment_ids,
                )
            )
            if trigger_msg is not None
            else 0
        )

        system_content = instructions
        budget = base_budget
        core_section = format_core_memories_section(
            effective_core_snapshot, max(0, budget - trigger_size)
        )
        if core_section:
            system_content += core_section
            budget -= len(core_section)

        if compacted_summary:
            summary_section = format_summary_section(
                compacted_summary, max(0, budget - trigger_size)
            )
            system_content += summary_section
            budget -= len(summary_section)

        system = (
            RunMessage(role="system", content=system_content)
            if system_content != instructions
            else base_system
        )

        history: list[RunMessage] = []
        included_message_ids: set[UUID] = set()
        for message in reversed(uncompacted):
            if message.role not in {"user", "assistant"}:
                continue
            turn = RunMessage(
                role=message.role,
                content=message.content,
                attachment_ids=message.attachment_ids
                if message.id == run.trigger_message_id
                else [],
            )
            size = entry_characters(turn)
            if size > budget or len(history) >= 97:
                if message.id == run.trigger_message_id:
                    raise ServiceError(
                        "The message for this response exceeds the context limit.", 422
                    )
                break
            history.append(turn)
            included_message_ids.add(message.id)
            budget -= size

        recalled_observation: RunObservation | None = None
        if (
            trigger_msg is not None
            and trigger_msg.role == "user"
            and hasattr(self.services, "list_resource_kinds")
            and extract_recall_terms(trigger_msg.content)
        ):
            if core_ids is None:
                active_memories = await fetch_active_memories(
                    self.services, run.agent_key
                )
                core_ids = {item.id for item in select_core_memories(active_memories)}
            recalled = await recall_archival_memories(
                self.services,
                run.agent_key,
                trigger_msg.content,
                exclude_ids=core_ids,
            )
            if recalled:
                recalled_observation = RunObservation(
                    name="recalled_memories",
                    data={
                        "note": "Relevant archival memories recalled for this turn.",
                        "memories": recalled,
                    },
                )

        recent_errors = await self.recent_run_errors(
            run, messages, included_message_ids
        )
        error_context = (
            RunObservation(
                name="recent_run_errors",
                data={
                    "notice": (
                        "Internal error context from recent responses in this conversation. "
                        "Do not share these details with the user unless they ask what happened."
                    ),
                    "errors": recent_errors,
                },
            )
            if recent_errors
            else None
        )
        context: list[TranscriptEntry] = [system, *reversed(history)]
        if (
            recalled_observation is not None
            and entry_characters(recalled_observation) <= budget
        ):
            context.append(recalled_observation)
            budget -= entry_characters(recalled_observation)
        if error_context is not None and entry_characters(error_context) <= budget:
            context.append(error_context)
        if event is not None:
            context.append(event)
        return context

    @traced("kadan.agent_runs.advance")
    async def advance(self, run: AgentRun) -> None:
        # Preparation plus two nodes per command.
        try:
            await self.graph.ainvoke(
                {"run": run}, {"recursion_limit": MAX_COMMANDS * 2 + 8}
            )
        except NodeCancelledError as error:
            # A cancelled boundary must leave the durable action recoverable,
            # just like cancellation of the enclosing task.
            raise asyncio.CancelledError() from error

    async def after_tool(self, state: RunState) -> str:
        return END if state.run.status == "completed" else "model"

    async def after_model(self, state: RunState) -> str:
        return "tool" if state.command is not None else "model"

    async def prepare(self, state: RunState) -> dict:
        run = state.run
        if not run.transcript:
            for attempt in range(MAX_ERROR_RETRIES + 1):
                if attempt > 0:
                    run = await self.owned(run)
                try:
                    status = await self.services.inference_status()
                    limit = self.transcript_limit(status, run)
                    run = await self.update(
                        run,
                        transcript=await self.context(
                            run, limit, inference_status=status
                        ),
                    )
                    await self._advance_group_chat_cursor(run)
                    break
                except (ServiceError, ValidationError, OSError) as error:
                    if (
                        is_lease_lost(error)
                        or (
                            isinstance(error, ServiceError)
                            and error.status in {409, 422}
                        )
                        or attempt >= MAX_ERROR_RETRIES
                    ):
                        raise
                    logger.warning(
                        "Agent context preparation failed (retry %d/%d) run=%s: %s",
                        attempt + 1,
                        MAX_ERROR_RETRIES,
                        run.id,
                        error,
                    )
        return {"run": run}

    @staticmethod
    def _retry_feedback_messages(
        error: ServiceError, retry_number: int
    ) -> list[BaseMessage]:
        call_id = f"retry_feedback_{retry_number}"
        payload = {
            "error": str(error),
            "retry": retry_number,
            "max_retries": MAX_ERROR_RETRIES,
            "instruction": (
                "Your previous step could not be executed. "
                "Respond with at most one valid tool call using only available tools, "
                "or send a concise plain text reply."
            ),
        }
        return [
            AIMessage(
                content="",
                tool_calls=[{"name": "step_retry_feedback", "args": {}, "id": call_id}],
            ),
            ToolMessage(
                content=json.dumps(payload, ensure_ascii=False),
                name="step_retry_feedback",
                tool_call_id=call_id,
            ),
        ]

    async def _infer_once(
        self,
        run: AgentRun,
        messages: list[BaseMessage],
        model_id: str | None,
        tools: list[StructuredTool],
    ) -> tuple[RunCall, ToolCommand, ArgumentRejection | None]:
        started = monotonic()
        logger.info(
            "Agent inference started run=%s at=%s",
            run.id,
            datetime.now(UTC).isoformat(),
        )
        draft_parts: list[str] = []
        draft_published = False
        last_draft_at = 0.0
        draft_path = self.path(run.agent_key, run.id) + "/draft"
        can_stream_draft = any(tool.name == "send_message" for tool in tools)

        async def publish_draft(text: str) -> None:
            nonlocal draft_published, last_draft_at
            if not can_stream_draft:
                return
            draft_parts.append(text)
            content = "".join(draft_parts)
            if not content.strip():
                return
            now = monotonic()
            if draft_published and now - last_draft_at < 0.15:
                return
            try:
                await self.services.json_request(
                    "Tsuchi",
                    "PUT",
                    draft_path,
                    json={"worker_id": str(self.owner), "content": content},
                )
                draft_published = True
                last_draft_at = now
            except ServiceError:
                logger.debug("Skipping transient draft update for run=%s", run.id)

        command: ToolCommand | None = None
        try:
            call = await self.inference.call(
                messages,
                model_id,
                f"kadan:{run.agent_key}:{run.conversation_id}",
                tools,
                priority="interactive",
                on_text_delta=publish_draft if can_stream_draft else None,
            )
            command = call.command
            rejection = validate(command, tools)
        finally:
            if draft_published and (command is None or command.tool != "send_message"):
                try:
                    await self.services.json_request(
                        "Tsuchi",
                        "PUT",
                        draft_path,
                        json={"worker_id": str(self.owner), "content": ""},
                    )
                except ServiceError:
                    logger.debug("Skipping transient draft clear for run=%s", run.id)
            logger.info(
                "Agent inference ended run=%s at=%s elapsed_seconds=%.3f",
                run.id,
                datetime.now(UTC).isoformat(),
                monotonic() - started,
            )
        return call, command, rejection

    @traced("kadan.agent_runs.infer_command")
    async def infer_command(self, state: RunState) -> dict:
        run = state.run
        run = await self.owned(run)
        if any(isinstance(item, UnrecoveredRecord) for item in run.transcript):
            raise ServiceError(
                "Kadan could not safely resume this saved response. Check any earlier actions before trying again.",
                422,
            )
        replayed = bool(run.transcript) and isinstance(run.transcript[-1], RunCall)
        if replayed:
            command = run.transcript[-1].command
        else:
            rejections = [
                item.data for item in run.transcript if item.role == "rejection"
            ]
            if len(rejections) > MAX_ARGUMENT_CORRECTIONS:
                rejection = ArgumentRejection.model_validate(rejections[-1])
                detail = "; ".join(
                    f"{problem.field}: {problem.code}" for problem in rejection.problems
                )
                raise ServiceError(
                    f"The agent exhausted its {MAX_ARGUMENT_CORRECTIONS} tool-argument "
                    f"correction attempts. Invalid {rejection.tool} arguments: {detail}. "
                    "Check the tool schema before trying again.",
                    502,
                    "invalid_command",
                )
            if (
                sum(item.role == "call" for item in run.transcript) + len(rejections)
                >= MAX_COMMANDS
            ):
                raise ServiceError("The agent reached its 24-command run limit.", 422)
            base_messages = transcript_messages(run.transcript)
            retry_messages: list[BaseMessage] = []
            call: RunCall | None = None
            command = None
            rejection = None
            for attempt in range(MAX_ERROR_RETRIES + 1):
                if attempt > 0:
                    run = await self.owned(run)
                try:
                    status = await self.services.inference_status()
                    if status.state != "ready":
                        raise ServiceError(status.message, 503)
                    mcp_servers = await self.active_mcp_servers()
                    tools = self.bound_tools(run, mcp_servers)
                    limit = self.transcript_limit(status, run, mcp_servers)
                    if sum(entry_characters(item) for item in run.transcript) > limit:
                        raise ServiceError(
                            "The agent run exceeded its context limit.", 422
                        )
                    call, command, rejection = await self._infer_once(
                        run,
                        [*base_messages, *retry_messages],
                        status.model_id,
                        tools,
                    )
                    break
                except (ServiceError, ValidationError, OSError) as error:
                    if (
                        is_lease_lost(error)
                        or (isinstance(error, ServiceError) and error.status == 422)
                        or attempt >= MAX_ERROR_RETRIES
                    ):
                        raise
                    logger.warning(
                        "Agent inference step failed (retry %d/%d) run=%s: %s",
                        attempt + 1,
                        MAX_ERROR_RETRIES,
                        run.id,
                        error,
                    )
                    if isinstance(error, ServiceError) and error.code in {
                        "invalid_command",
                        "response_limit",
                        "command_too_large",
                    }:
                        retry_messages = self._retry_feedback_messages(
                            error, attempt + 1
                        )
                    else:
                        retry_messages = []
            assert call is not None and command is not None
            if rejection is not None:
                # A rejection is feedback, never a pending assistant action. Save
                # it atomically so restart cannot replay it or reset the budget.
                run = await self.update(
                    run,
                    transcript=[
                        *run.transcript,
                        RunObservation(
                            role="rejection",
                            name=command.tool,
                            data=rejection.model_dump(mode="json"),
                        ),
                    ],
                )
                return {"run": run, "command": None, "replayed": False}
        if not replayed:
            # Persist the action before side effects. Recovery reuses its derived ID.
            run = await self.update(
                run,
                transcript=[
                    *run.transcript,
                    call,
                ],
            )
        return {"run": run, "command": command, "replayed": replayed}

    async def active_mcp_servers(self) -> list[McpServerRecord]:
        mcp = getattr(self.services, "mcp", None)
        if mcp is None:
            return []
        try:
            servers = await mcp.list_servers()
        except ServiceError:
            return []
        return [
            server
            for server in servers
            if server.enabled and server.status == "connected"
        ]

    def transcript_limit(
        self,
        status,
        run: AgentRun,
        mcp_servers: list[McpServerRecord] | None = None,
    ) -> int:
        """The transcript budget left after the bound tool definitions."""
        limit = context_characters(
            getattr(status, "context_limit", None)
        ) - tool_characters(self.bound_tools(run, mcp_servers))
        if limit <= 0:
            raise ServiceError(
                "The model context limit leaves no room for agent instructions "
                "after reserving tools and a reply.",
                422,
            )
        return limit

    async def _build_group_chat_event_data(self, run: AgentRun, detail: object) -> dict:
        group_chat = getattr(self.services, "group_chat", None)
        base: dict = dict(detail) if isinstance(detail, dict) else {"detail": detail}
        if group_chat is None:
            return base
        try:
            room_id = (
                UUID(str(base["room_id"]))
                if isinstance(base.get("room_id"), str)
                else (await group_chat.default_room()).id
            )
            participant = await group_chat.ensure_kadan_participant(
                room_id, agent_key=run.agent_key
            )
            if participant is None:
                return base
            page = await group_chat.list_messages(
                room_id, after_seq=participant.cursor_seq, limit=100
            )
            roster = await group_chat.list_participants(room_id)
            by_id = {item.id: item for item in roster}
            unread = [
                GroupChatMcpHandler.format_message(
                    msg, participant=participant, participants_by_id=by_id
                )
                for msg in page.messages
            ]
            addressed_by = [
                {
                    "seq": msg["seq"],
                    "sender_name": msg["sender_name"],
                    "sender_kind": msg["sender_kind"],
                }
                for msg in unread
                if msg["addressed_to_me"]
            ]
            last_included_seq = (
                page.messages[-1].seq if page.messages else participant.cursor_seq
            )
            return {
                "room_id": str(page.room.id),
                "room_title": page.room.title,
                "my_participant_id": str(participant.id),
                "my_handle": participant.handle,
                "my_display_name": participant.display_name,
                "cursor_seq": participant.cursor_seq,
                "last_included_seq": last_included_seq,
                "latest_seq": page.latest_seq,
                "has_more": page.has_more,
                "trigger_seq": base.get("trigger_seq"),
                "addressed_by": addressed_by,
                "unread_messages": unread,
                "participants": [
                    {
                        "id": str(item.id),
                        "handle": item.handle,
                        "display_name": item.display_name,
                        "kind": item.kind,
                        "description": item.description,
                        "active": item.enabled and item.revoked_at is None,
                    }
                    for item in roster
                ],
                "consecutive_agent_turns": page.room.consecutive_agent_turns,
                "max_consecutive_agent_turns": page.room.max_consecutive_agent_turns,
                "turn_cap_reached": (
                    page.room.consecutive_agent_turns
                    >= page.room.max_consecutive_agent_turns
                ),
                "instruction": (
                    "You were woken for Group Chat. Reply in Group Chat using "
                    "group_chat_send_message (pass handoff_to only to hand the turn "
                    "to another agent, omit handoff_to to return the floor to the user), "
                    "or call finish silently if no reply is needed."
                ),
            }
        except (ServiceError, ValueError, TypeError) as error:
            logger.warning(
                "Could not hydrate full Group Chat context for run=%s: %s",
                run.id,
                error,
            )
            return base

    async def _advance_group_chat_cursor(self, run: AgentRun) -> None:
        if not is_group_chat_run(run):
            return
        group_chat = getattr(self.services, "group_chat", None)
        if group_chat is None:
            return
        obs = next(
            (
                item
                for item in reversed(run.transcript)
                if isinstance(item, RunObservation) and item.name == "group_chat_event"
            ),
            None,
        )
        if obs is None or not isinstance(obs.data, dict):
            return
        room_raw = obs.data.get("room_id")
        pid_raw = obs.data.get("my_participant_id")
        consumed_seq = obs.data.get("last_included_seq")
        if not isinstance(consumed_seq, int) or isinstance(consumed_seq, bool):
            unread_items = obs.data.get("unread_messages")
            if isinstance(unread_items, list) and unread_items:
                seqs = [
                    item["seq"]
                    for item in unread_items
                    if isinstance(item, dict)
                    and isinstance(item.get("seq"), int)
                    and not isinstance(item.get("seq"), bool)
                ]
                consumed_seq = max(seqs) if seqs else 0
            else:
                consumed_seq = 0
        if (
            not isinstance(room_raw, str)
            or not isinstance(pid_raw, str)
            or consumed_seq <= 0
        ):
            return
        try:
            await group_chat.touch_participant(
                UUID(room_raw),
                UUID(pid_raw),
                last_seen_at=datetime.now(UTC),
                cursor_seq=consumed_seq,
            )
        except (ServiceError, ValueError):
            logger.debug(
                "Skipping transient Group Chat cursor update for run=%s", run.id
            )

    def bound_tools(
        self,
        run: AgentRun,
        mcp_servers: list[McpServerRecord] | None = None,
    ) -> list[StructuredTool]:
        if is_group_chat_run(run):
            base = [tool for tool in ALL_TOOLS if tool.name != "send_message"] + list(
                GROUP_CHAT_TOOLS
            )
        else:
            base = list(ALL_TOOLS)
        if mcp_servers:
            mcp_tools, _ = build_mcp_tools(mcp_servers, {t.name for t in base})
            return [*base, *mcp_tools]
        return base

    @traced("kadan.agent_runs.invoke_tool")
    async def invoke_tool(self, state: RunState) -> dict:
        command = state.command
        assert command is not None
        run = await self.owned(state.run)
        pending = run.transcript[-1] if run.transcript else None
        if not isinstance(pending, RunCall) or pending.command != command:
            raise ServiceError(
                "The selected action does not match the saved work.", 409
            )
        tool = TOOLS_BY_NAME.get(command.tool) or GROUP_CHAT_TOOLS_BY_NAME.get(
            command.tool
        )
        conversation = RunConversation(self, run, state.replayed)
        mcp_target: tuple[McpServerRecord, Any] | None = None
        if tool is None:
            mcp_servers = await self.active_mcp_servers()
            gateway_by_name = {t.name: t for t in MCP_GATEWAY_TOOLS}
            if command.tool in gateway_by_name:
                tool = gateway_by_name[command.tool]
            else:
                _, registry = build_mcp_tools(
                    mcp_servers,
                    set(TOOLS_BY_NAME) | set(GROUP_CHAT_TOOLS_BY_NAME),
                )
                mcp_target = registry.get(command.tool)

        if mcp_target is not None:
            server, tool_info = mcp_target
            result = await execute_mcp_tool(
                self.services,
                server,
                tool_info,
                command.arguments,
                action_id=conversation.action_id,
                replayed=state.replayed,
                origin=McpOrigin(
                    agent_key=run.agent_key,
                    conversation_id=run.conversation_id,
                    run_id=run.id,
                ),
            )
        elif tool is None:
            result = {"tool": command.tool, "error": "This tool no longer exists."}
        elif tool in MCP_GATEWAY_TOOLS or tool in GROUP_CHAT_TOOLS:
            result = await tool.ainvoke(
                command.arguments,
                config={"configurable": {"conversation": conversation}},
            )
        elif tool in PERSONA_TOOLS:
            if state.replayed and command.tool == "update_persona":
                result = {
                    "error": "The runner restarted before this persona save reported "
                    "a result. It may or may not have saved; read_persona before retrying."
                }
            elif state.replayed and command.tool == "rename_self":
                result = {
                    "error": "The runner restarted before this rename reported "
                    "a result. It may or may not have saved; check before retrying."
                }
            else:
                try:
                    result = await tool.ainvoke(
                        command.arguments,
                        config={"configurable": {"conversation": conversation}},
                    )
                except (ServiceError, ValidationError) as error:
                    if is_lease_lost(error):
                        raise
                    result = {"error": str(error)}

        elif tool in MEMORY_TOOLS:
            result = await self.use_memory(tool, command, run, state.replayed)
        elif tool in CONVERSATION_TOOLS:
            result = await tool.ainvoke(
                command.arguments,
                config={"configurable": {"conversation": conversation}},
            )
        else:
            result = {"tool": command.tool, "error": "This tool is not available."}
        if conversation.finished:
            run = await self.update(
                run, status="completed", owner_id=None, lease_expires_at=None
            )
            self._wake_memory_queue()
        else:
            run = await self.save_result(
                run, command.tool, result, conversation.end_run
            )
        if conversation.post_save_action is not None:
            try:
                await conversation.post_save_action()
            except Exception:
                logger.exception(
                    "Post-save action failed for run=%s tool=%s",
                    run.id,
                    command.tool,
                )
        return {"run": run, "command": None, "replayed": False}

    async def use_memory(
        self, tool: StructuredTool, command: ToolCommand, run: AgentRun, replayed: bool
    ) -> dict:
        if replayed and tool.metadata["method"] != "GET":
            return {
                "error": "The runner restarted before this memory write reported "
                "a result. It may or may not have saved; read or search your memories "
                "before retrying. For forget_memory, use read_memory with "
                "include_trashed to check its trash state."
            }
        try:
            return await tool.ainvoke(
                command.arguments,
                config={
                    "configurable": {
                        "memories": AgentMemories(
                            self.services, run.agent_key, lambda: self.owned(run)
                        )
                    }
                },
            )
        except (ServiceError, ValidationError) as error:
            return {"error": str(error)}

    def _wake_memory_queue(self) -> None:
        memory_queue = getattr(self.services, "memory_queue", None)
        if memory_queue is not None:
            memory_queue.wakeup.set()

    def _preempt_memory_queue(self) -> None:
        memory_queue = getattr(self.services, "memory_queue", None)
        if memory_queue is not None and hasattr(memory_queue, "preempt"):
            memory_queue.preempt()

    async def save_result(
        self, run: AgentRun, name: str, result: dict, end_run: bool
    ) -> AgentRun:
        # Save the result and release ownership together. Recovery before this
        # write replays the same action ID; after it no inference is scheduled.
        run = await self.update(
            run,
            transcript=[
                *run.transcript,
                RunResult(call_id=run.transcript[-1].call_id, data=result),
            ],
            **(
                {"status": "completed", "owner_id": None, "lease_expires_at": None}
                if end_run
                else {}
            ),
        )
        if end_run:
            logger.info(
                "Agent run yielded run=%s tool=%s at=%s",
                run.id,
                name,
                datetime.now(UTC).isoformat(),
            )
            self._wake_memory_queue()
        return run
