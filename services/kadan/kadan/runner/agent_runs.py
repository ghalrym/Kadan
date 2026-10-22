"""Conversation work resumes on durable user or image events, never idle inference."""

from __future__ import annotations

import asyncio
import json
import logging
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
from ..api.pydantic_models.agent_runs import (
    AgentRun,
    AgentRunState,
    ImageJob,
    RunMessage,
)
from ..api.pydantic_models.resources import Message, ResourceKind
from ..api.run_client import RunClient
from ..api.services import parse_resource
from .agent_inference import (
    AgentInference,
    ToolCommand,
    parse_command,
    tool_characters,
)
from .tools import ALL_TOOLS, CONVERSATION_TOOLS, TOOLS_BY_NAME
from .tools.conversation import GenerateImageArguments, SendMessageArguments
from .tools.workspace import (
    CreateProjectArguments,
    CreateWorkspaceDocumentArguments,
    ListWorkspaceDocumentsArguments,
    OpenWorkspaceFileArguments,
    ReadWorkspaceDocumentArguments,
    ReadWorkspaceFileArguments,
    SaveWorkspaceFileArguments,
    UpdateProjectArguments,
    UpdateWorkspaceDocumentArguments,
)
from .workspace import AgentWorkspace, ProjectStatus, document_content
from .workspace_files import WorkspaceFiles

if TYPE_CHECKING:
    from ..api.services import LocalServices

logger = logging.getLogger(__name__)
LEASE_SECONDS = 60
MAX_COMMANDS = 24
# Transcript budget in characters when Hikari does not report a context window.
MAX_CONTEXT = 48000
# Tool results are JSON, which tokenizes densely; three characters per token
# keeps a transcript inside the window. The reserve covers the reply and template.
CHARACTERS_PER_TOKEN = 3
RESERVED_TOKENS = 1024
TOOL_RESULT = "Tool result: "


def context_characters(context_limit: int | None) -> int:
    """The agent's transcript budget, sized to the selected model's window."""
    if context_limit is None:
        return MAX_CONTEXT
    return max(0, (context_limit - RESERVED_TOKENS) * CHARACTERS_PER_TOKEN)


def transcript_messages(transcript: list[RunMessage]) -> list[BaseMessage]:
    """The saved transcript as LangChain messages, with tool calls and results."""
    messages: list[BaseMessage] = []
    call = 0
    for item in transcript:
        if item.role == "system":
            messages.append(SystemMessage(content=item.content))
        elif item.role == "assistant" and item.content.startswith('{"tool":'):
            command = parse_command(item.content)
            call += 1
            messages.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": command.tool,
                            "args": command.arguments,
                            "id": f"call_{call}",
                        }
                    ],
                )
            )
        elif item.role == "assistant":
            messages.append(AIMessage(content=item.content))
        elif item.content.startswith(TOOL_RESULT) and call:
            messages.append(
                ToolMessage(
                    content=item.content.removeprefix(TOOL_RESULT),
                    tool_call_id=f"call_{call}",
                )
            )
        else:
            messages.append(HumanMessage(content=item.content))
    return messages


def validate(command: ToolCommand, tools: list[StructuredTool]) -> None:
    """Reject unknown tools and invalid arguments before the call is saved."""
    if command.tool not in {tool.name for tool in tools}:
        raise ServiceError(f"unknown tool {command.tool!r}", 502, "invalid_command")
    try:
        TOOLS_BY_NAME[command.tool].args_schema.model_validate(command.arguments)
    except ValidationError as error:
        raise ServiceError(
            f"invalid {command.tool} arguments: {error.errors()[0]['msg']}",
            502,
            "invalid_command",
        ) from error


class RunConversation:
    """The conversation tools' view of one run, at the step being executed."""

    def __init__(self, runs: AgentRuns, run: AgentRun, replayed: bool = False):
        self.runs = runs
        self.run = run
        self.action_id = uuid5(run.id, f"command:{len(run.transcript)}")
        self.end_run = False
        self.finished = False
        self.replayed = replayed

    async def send_message(self, args: SendMessageArguments) -> dict:
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
        return {
            "tool": "send_message",
            "delivered": True,
            "message_id": str(self.action_id),
        }

    async def generate_image(self, args: GenerateImageArguments) -> dict:
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

    async def finish(self) -> None:
        self.finished = True

    async def save_workspace_file(self, args: SaveWorkspaceFileArguments) -> dict:
        result = {"tool": "save_workspace_file", "file_id": str(self.action_id)}
        files = WorkspaceFiles(self.runs.services, self.run.agent_key)
        try:
            saved = await files.save(
                self.action_id,
                args.path,
                args.title,
                lambda: self.runs.owned(self.run),
                replayed=self.replayed,
            )
        except ServiceError as error:
            return {**result, "error": str(error)}
        except OSError as error:
            return {
                **result,
                "error": "The file transfer could not be confirmed. Check this file_id "
                "with read_workspace_file before retrying. " + str(error),
            }
        return {**result, **saved}

    async def open_workspace_file(self, args: OpenWorkspaceFileArguments) -> dict:
        result = {"tool": "open_workspace_file", "file_id": str(args.file_id)}
        try:
            opened = await WorkspaceFiles(self.runs.services, self.run.agent_key).open(
                args.file_id,
                self.action_id,
                lambda: self.runs.owned(self.run),
                replayed=self.replayed,
            )
        except (ServiceError, OSError) as error:
            return {**result, "error": str(error)}
        return {**result, **opened}

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

    async def workspace_operation(
        self,
        name: str,
        operation: Callable[[AgentWorkspace], Awaitable[dict]],
        *,
        writing: bool = False,
    ) -> dict:
        result = {"tool": name}
        if writing and self.replayed:
            return {
                **result,
                "error": "The runner restarted before this write reported a result. It may have been saved; read Workspace before retrying.",
            }
        workspace = AgentWorkspace(self.runs.services, self.run.agent_key)
        async with self.runs.lock:
            await self.runs.owned(self.run)
            try:
                value = await operation(workspace)
            except ServiceError as error:
                message = str(error)
                if writing and error.status >= 500:
                    message += " The write may have been saved; read Workspace before retrying."
                return {**result, "error": message}
        return {**result, **value}

    async def list_workspace_documents(
        self, args: ListWorkspaceDocumentsArguments
    ) -> dict:
        return await self.workspace_operation(
            "list_workspace_documents",
            lambda workspace: workspace.list(
                query=args.query,
                limit=args.limit,
                offset=args.offset,
                projects_only=args.projects_only,
            ),
        )

    async def read_workspace_document(
        self, args: ReadWorkspaceDocumentArguments
    ) -> dict:
        async def read(workspace: AgentWorkspace) -> dict:
            return document_content(await workspace.get(args.document_id))

        return await self.workspace_operation("read_workspace_document", read)

    async def create_project(self, args: CreateProjectArguments) -> dict:
        return await self.workspace_operation(
            "create_project",
            lambda workspace: workspace.create_project(args.title, args.markdown),
            writing=True,
        )

    async def update_workspace_document(
        self, args: UpdateWorkspaceDocumentArguments
    ) -> dict:
        return await self.update_document(args)

    async def update_project(self, args: UpdateProjectArguments) -> dict:
        return await self.update_document(args, args.status)

    async def update_document(
        self,
        args: UpdateWorkspaceDocumentArguments,
        status: ProjectStatus | None = None,
    ) -> dict:
        async def write(workspace: AgentWorkspace) -> dict:
            current = await workspace.get(args.document_id)
            # Storage reads may wait; fence again immediately before the write.
            await self.runs.owned(self.run)
            return await workspace.update(
                current, args.expected_version, args.markdown, status
            )

        return await self.workspace_operation(
            "update_project" if status is not None else "update_workspace_document",
            write,
            writing=True,
        )

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
        graph.add_edge("model", "tool")
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
                "expected_version": run.version,
                "worker_id": str(self.owner),
            },
        )
        return AgentRun.model_validate(value)

    async def owned(self, run: AgentRun) -> AgentRun:
        current = await self.get(run.agent_key, run.id)
        if (
            current.status != "running"
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
        logger.info(
            "Agent run claimed run=%s at=%s", run.id, datetime.now(UTC).isoformat()
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
            try:
                await self.update(
                    run,
                    status="failed",
                    owner_id=None,
                    lease_expires_at=None,
                    error=str(error)[:2000],
                )
            except ServiceError:
                logger.info(
                    "Run ended after cancellation or loss of ownership: %s", run.id
                )
        finally:
            work.cancel()
            heartbeat.cancel()
            await asyncio.gather(work, heartbeat, return_exceptions=True)

    @traced("kadan.agent_runs.context")
    async def context(
        self, run: AgentRun, limit: int | None = None
    ) -> list[RunMessage]:
        instructions = await asyncio.to_thread(
            Path(self.services.settings.agent_instruction_file).read_text
        )
        conversation = await self.services.resource(run.agent_key, run.conversation_id)
        if conversation.kind != ResourceKind.CONVERSATION or conversation.deleted_at:
            raise ServiceError("The conversation is no longer available.", 409)
        if run.persona_id:
            persona = await self.services.resource(run.agent_key, run.persona_id)
            if persona.kind != ResourceKind.PERSONA or persona.deleted_at:
                raise ServiceError("The selected persona is no longer available.", 409)
            instructions += "\nSelected persona:\n" + persona.markdown
        system = RunMessage(role="system", content=instructions)
        if run.trigger_job_id:
            image = ImageJob.model_validate(
                await self.services.json_request(
                    "Tsuchi",
                    "GET",
                    f"/api/agents/{run.agent_key}/image-jobs/{run.trigger_job_id}",
                )
            )
            trigger = RunMessage(
                role="user",
                content="Image job event: "
                + json.dumps(
                    {
                        "job_id": str(image.id),
                        "status": image.status,
                        "prompt": image.prompt,
                        "error": image.error,
                        "delivered": image.delivered_message_id is not None,
                    }
                ),
            )
        else:
            trigger = RunMessage(
                role="user",
                content=f"Respond to user message {run.trigger_message_id} in the conversation above. Other user messages have their own queued runs.",
            )
        jobs = [
            ImageJob.model_validate(value)
            for value in await self.services.json_request(
                "Tsuchi",
                "GET",
                f"/api/agents/{run.agent_key}/resources/{run.conversation_id}/image-jobs",
            )
        ]
        # Pending jobs survive their submitting run; later turns must see them.
        pending = [
            job for job in jobs if job.status in {"queued", "running", "cancelling"}
        ]
        if pending:
            trigger.content += "\nPending image jobs (do not resubmit): " + json.dumps(
                [
                    {"job_id": str(job.id), "status": job.status, "prompt": job.prompt}
                    for job in pending
                ]
            )
        try:
            projects = await AgentWorkspace(self.services, run.agent_key).list(
                limit=8, projects_only=True
            )
            if projects["documents"] or projects["next_offset"] is not None:
                trigger.content += (
                    "\nWorkspace project index (saved plans, not running jobs). Read relevant plans before acting; do not resume waiting or terminal projects without explicit user direction. Use list_workspace_documents(projects_only=true, offset=next_offset) for more: "
                    + json.dumps(projects)
                )
        except ServiceError as error:
            trigger.content += (
                "\nWorkspace project index unavailable: "
                + str(error)
                + ". Do not assume there are no existing projects or create a duplicate; check Workspace first."
            )
        if limit is None:
            limit = MAX_CONTEXT - tool_characters(self.bound_tools())
        budget = limit - len(system.content) - len(trigger.content)
        if budget < 0:
            raise ServiceError(
                "Agent instructions and event exceed the context limit.", 422
            )
        history = []
        for message in reversed(conversation.messages):
            if message.role not in {"user", "assistant"}:
                continue
            text = f"[message {message.id}] " + message.content
            if message.attachment_ids:
                text += (
                    " [Attached assets: "
                    + ", ".join(map(str, message.attachment_ids))
                    + "]"
                )
            if len(text) > budget or len(history) >= 97:
                break
            history.append(RunMessage(role=message.role, content=text))
            budget -= len(text)
        if run.trigger_message_id and not any(
            str(run.trigger_message_id) in item.content for item in history
        ):
            raise ServiceError(
                "The triggering message is outside the agent context window.", 422
            )
        return [system, *reversed(history), trigger]

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

    async def prepare(self, state: RunState) -> dict:
        run = state.run
        if not run.transcript:
            status = await self.services.inference_status()
            limit = self.transcript_limit(status)
            run = await self.update(run, transcript=await self.context(run, limit))
        return {"run": run}

    @traced("kadan.agent_runs.infer_command")
    async def infer_command(self, state: RunState) -> dict:
        run = state.run
        run = await self.owned(run)
        replayed = run.transcript[-1].role == "assistant"
        if replayed:
            command = parse_command(run.transcript[-1].content)
        else:
            if (
                sum(
                    item.role == "assistant" and item.content.startswith('{"tool":')
                    for item in run.transcript
                )
                >= MAX_COMMANDS
            ):
                raise ServiceError("The agent reached its 24-command run limit.", 422)
            status = await self.services.inference_status()
            if status.state != "ready":
                raise ServiceError(status.message, 503)
            limit = self.transcript_limit(status)
            if sum(len(item.content) for item in run.transcript) > limit:
                raise ServiceError("The agent run exceeded its context limit.", 422)
            started = monotonic()
            logger.info(
                "Agent inference started run=%s at=%s",
                run.id,
                datetime.now(UTC).isoformat(),
            )
            try:
                tools = self.bound_tools()
                command = await self.inference.call(
                    transcript_messages(run.transcript),
                    status.model_id,
                    f"kadan:{run.agent_key}:{run.conversation_id}",
                    tools,
                )
                validate(command, tools)
            finally:
                logger.info(
                    "Agent inference ended run=%s at=%s elapsed_seconds=%.3f",
                    run.id,
                    datetime.now(UTC).isoformat(),
                    monotonic() - started,
                )
            # Persist the action before side effects. Recovery reuses its derived ID.
            run = await self.update(
                run,
                transcript=[
                    *run.transcript,
                    RunMessage(role="assistant", content=command.model_dump_json()),
                ],
            )
        return {"run": run, "command": command, "replayed": replayed}

    def transcript_limit(self, status) -> int:
        """The transcript budget left after the bound tool definitions."""
        limit = context_characters(
            getattr(status, "context_limit", None)
        ) - tool_characters(self.bound_tools())
        if limit <= 0:
            raise ServiceError(
                "The model context limit leaves no room for agent instructions "
                "after reserving tools and a reply.",
                422,
            )
        return limit

    def bound_tools(self) -> list[StructuredTool]:
        """All tools; the computer's only when one is configured."""
        return ALL_TOOLS if self.services.sandbox.configured else CONVERSATION_TOOLS

    @traced("kadan.agent_runs.invoke_tool")
    async def invoke_tool(self, state: RunState) -> dict:
        command = state.command
        assert command is not None
        run = state.run
        tool = TOOLS_BY_NAME.get(command.tool)
        conversation = RunConversation(self, run, state.replayed)
        if tool is None:
            result = {"tool": command.tool, "error": "This tool no longer exists."}
        elif tool in CONVERSATION_TOOLS:
            result = await tool.ainvoke(
                command.arguments,
                config={"configurable": {"conversation": conversation}},
            )
        else:
            result = {
                "tool": tool.name,
                **await self.use_computer(tool, command, state.replayed),
            }
        if conversation.finished:
            run = await self.update(
                run, status="completed", owner_id=None, lease_expires_at=None
            )
        else:
            run = await self.save_result(
                run, command.tool, result, conversation.end_run
            )
        return {"run": run, "command": None, "replayed": False}

    async def use_computer(
        self, tool: StructuredTool, command: ToolCommand, replayed: bool
    ) -> dict:
        if not self.services.sandbox.configured:
            return {"error": "No computer is configured for this agent."}
        if replayed and tool.metadata["method"] != "GET":
            # The sandbox has no idempotency keys, so a restart must not
            # repeat a side effect that may already have happened.
            return {
                "error": "The runner restarted before this operation reported "
                "a result. It may or may not have run; check before retrying."
            }
        try:
            return await tool.ainvoke(
                command.arguments,
                config={"configurable": {"computer": self.services.sandbox}},
            )
        except (ServiceError, ValidationError) as error:
            return {"error": str(error)}

    async def save_result(
        self, run: AgentRun, name: str, result: dict, end_run: bool
    ) -> AgentRun:
        # Save the result and release ownership together. Recovery before this
        # write replays the same action ID; after it no inference is scheduled.
        run = await self.update(
            run,
            transcript=[
                *run.transcript,
                RunMessage(role="user", content=TOOL_RESULT + json.dumps(result)),
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
        return run
