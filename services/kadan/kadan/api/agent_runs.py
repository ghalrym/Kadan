"""Conversation work resumes on durable user or image events, never idle inference."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING
from uuid import UUID, uuid4, uuid5

from langchain_core.tools import StructuredTool
from langgraph.errors import NodeCancelledError
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from .agent_inference import (
    TOOL_PROTOCOL,
    AgentCommand,
    AgentInference,
    Finish,
    GenerateImage,
    parse_command,
)
from .config import ServiceError
from .pydantic_models.agent_runs import AgentRun, AgentRunState, ImageJob, RunMessage
from .pydantic_models.inference import CompletionMessage
from .pydantic_models.resources import Message, ResourceKind

if TYPE_CHECKING:
    from .services import LocalServices

logger = logging.getLogger(__name__)
LEASE_SECONDS = 60
MAX_COMMANDS = 24
MAX_CONTEXT = 48000


class RunState(BaseModel):
    run: AgentRun
    command: AgentCommand | None = None
    corrections: int = 0


class AgentRuns:
    def __init__(self, services: LocalServices):
        self.services = services
        self.owner = uuid4()
        self.wakeup = asyncio.Event()
        self.runner: asyncio.Task | None = None
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

    def start(self) -> None:
        self.runner = asyncio.create_task(self.run(), name="agent-runs")

    async def close(self) -> None:
        if self.runner:
            self.runner.cancel()
            await asyncio.gather(self.runner, return_exceptions=True)

    @staticmethod
    def path(key: str, run_id: UUID) -> str:
        return f"/api/agents/{key}/runs/{run_id}"

    async def get(self, key: str, run_id: UUID) -> AgentRun:
        run = AgentRun.model_validate(
            await self.services.json_request("Tsuchi", "GET", self.path(key, run_id))
        )
        if run.agent_key != key or run.id != run_id:
            raise ServiceError("Tsuchi returned the wrong agent run.", 502)
        return run

    async def enqueue(self, key: str, conversation_id: UUID, value) -> AgentRun:
        # Storage checks idempotent retries before checking the conversation version.
        message = Message(
            id=value.message_id,
            role="user",
            content=value.content,
            resource_ids=[value.persona_id] if value.persona_id else [],
        )
        run = AgentRun.model_validate(
            await self.services.json_request(
                "Tsuchi",
                "POST",
                f"/api/agents/{key}/resources/{conversation_id}/runs",
                json={
                    "message": message.model_dump(mode="json"),
                    "expected_version": value.expected_version,
                    "persona_id": str(value.persona_id) if value.persona_id else None,
                },
            )
        )
        self.wakeup.set()
        return run

    async def cancel(self, key: str, run_id: UUID) -> AgentRun:
        result = await self.services.json_request(
            "Tsuchi", "POST", self.path(key, run_id) + "/cancel", json={}
        )
        self.wakeup.set()
        return AgentRun.model_validate(result)

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

    async def run(self) -> None:
        while True:
            self.wakeup.clear()
            try:
                values = await self.services.json_request(
                    "Tsuchi", "GET", "/api/agent-runs", params={"active": "true"}
                )
                for value in values:
                    run = AgentRun.model_validate(value)
                    if run.status == "queued" or (
                        run.status == "running"
                        and run.lease_expires_at <= datetime.now(UTC)
                    ):
                        await self.process(run)
            except (ServiceError, ValidationError):
                logger.exception("Could not process the durable agent queue")
            try:
                await asyncio.wait_for(self.wakeup.wait(), timeout=2)
            except TimeoutError:
                pass

    async def heartbeat(self, run: AgentRun) -> None:
        while True:
            await asyncio.sleep(10)
            await self.update(
                run,
                lease_expires_at=datetime.now(UTC) + timedelta(seconds=LEASE_SECONDS),
            )

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
        heartbeat = asyncio.create_task(self.heartbeat(run))
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

    async def context(self, run: AgentRun) -> list[RunMessage]:
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
        system = RunMessage(role="system", content=instructions + "\n" + TOOL_PROTOCOL)
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
        budget = MAX_CONTEXT - len(system.content) - len(trigger.content)
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

    async def advance(self, run: AgentRun) -> None:
        # At most three invalid-command attempts plus two nodes per command.
        try:
            await self.graph.ainvoke(
                {"run": run}, {"recursion_limit": MAX_COMMANDS * 2 + 8}
            )
        except NodeCancelledError as error:
            # A cancelled boundary must leave the durable action recoverable,
            # just like cancellation of the enclosing task.
            raise asyncio.CancelledError() from error

    async def after_model(self, state: RunState) -> str:
        return "tool" if state.command is not None else "model"

    async def after_tool(self, state: RunState) -> str:
        return END if state.run.status == "completed" else "model"

    async def prepare(self, state: RunState) -> dict:
        run = state.run
        if not run.transcript:
            run = await self.update(run, transcript=await self.context(run))
        return {"run": run}

    async def infer_command(self, state: RunState) -> dict:
        run = state.run
        run = await self.owned(run)
        if run.transcript[-1].role == "assistant":
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
            if sum(len(item.content) for item in run.transcript) > MAX_CONTEXT:
                raise ServiceError("The agent run exceeded its context limit.", 422)
            status = await self.services.inference_status()
            if status.state != "ready":
                raise ServiceError(status.message, 503)
            started = monotonic()
            logger.info(
                "Agent inference started run=%s at=%s",
                run.id,
                datetime.now(UTC).isoformat(),
            )
            try:
                command = await self.inference.call(
                    [CompletionMessage(**item.model_dump()) for item in run.transcript],
                    status.model_id,
                    f"kadan:{run.agent_key}:{run.conversation_id}",
                )
            except ServiceError as error:
                if error.code != "invalid_command" or state.corrections >= 2:
                    raise
                run = await self.update(
                    run,
                    transcript=[
                        *run.transcript,
                        RunMessage(
                            role="user",
                            content="Your response was not a valid command. Return exactly one JSON object using the documented tool schema. No user message was delivered.",
                        ),
                    ],
                )
                return {
                    "run": run,
                    "command": None,
                    "corrections": state.corrections + 1,
                }
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
        return {"run": run, "command": command}

    async def invoke_tool(self, state: RunState) -> dict:
        command = state.command
        assert command is not None

        async def execute(**arguments):
            return await self.execute_command(state.run, type(command)(**arguments))

        tool = StructuredTool.from_function(
            coroutine=execute,
            name=command.tool,
            description=f"Execute the persisted {command.tool} command.",
            args_schema=type(command),
        )
        run = await tool.ainvoke(command.model_dump())
        return {"run": run, "command": None}

    async def execute_command(self, run: AgentRun, command: AgentCommand) -> AgentRun:
        if isinstance(command, Finish):
            return await self.update(
                run, status="completed", owner_id=None, lease_expires_at=None
            )
        action_id = uuid5(run.id, f"command:{len(run.transcript)}")
        if isinstance(command, GenerateImage):
            image = ImageJob.model_validate(
                await self.services.json_request(
                    "Tsuchi",
                    "POST",
                    self.path(run.agent_key, run.id) + "/image-jobs",
                    json={
                        "id": str(action_id),
                        "prompt": command.prompt,
                        "request_id": str(uuid5(action_id, "request")),
                        "worker_id": str(self.owner),
                    },
                )
            )
            self.services.images.wakeup.set()
            result = {
                "tool": "generate_image",
                "job_id": str(image.id),
                "status": image.status,
                "delivered": image.delivered_message_id is not None,
            }
        else:
            message = Message(id=action_id, role="assistant", content=command.content)
            async with self.lock:
                await self.services.json_request(
                    "Tsuchi",
                    "POST",
                    self.path(run.agent_key, run.id) + "/messages",
                    json={
                        "worker_id": str(self.owner),
                        "message": message.model_dump(mode="json"),
                        "image_job_id": str(command.image_job_id)
                        if command.image_job_id
                        else None,
                    },
                )
            result = {
                "tool": "send_message",
                "delivered": True,
                "message_id": str(action_id),
            }
        end_run = isinstance(command, GenerateImage) or command.end_run
        # Save the result and release ownership together. Recovery before this
        # write replays the same action ID; after it no inference is scheduled.
        run = await self.update(
            run,
            transcript=[
                *run.transcript,
                RunMessage(role="user", content="Tool result: " + json.dumps(result)),
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
                command.tool,
                datetime.now(UTC).isoformat(),
            )
        return run
