"""The computer agent: one long-lived thread per agent, outside conversation runs.

Conversation runs hand work over with `use_computer` and end immediately. This
loop claims each agent's thread, works through its queued tasks in order with
the computer tools, folds in user steering before every inference, and reports
each result back to the conversation that asked through Tsuchi.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING
from uuid import uuid4

from kadan_tracing import traced

from ..api.config import ServiceError
from ..api.pydantic_models.agent_runs import (
    RunCall,
    RunMessage,
    RunObservation,
    RunResult,
    TaskEnd,
    TaskHistory,
    TaskOutcome,
    TaskStart,
    ToolCommand,
    TranscriptEntry,
    UnrecoveredRecord,
)
from ..api.pydantic_models.computer import ComputerTask, ComputerThread
from .agent_inference import AgentInference, tool_characters
from .agent_runs import (
    context_characters,
    validate,
)
from .tools import COMPUTER_AGENT_TOOLS, TOOLS_BY_NAME
from .tools.base import invoke_computer_tool
from .tools.computer import FinishTaskArguments
from .transcript import entry_characters, transcript_messages

if TYPE_CHECKING:
    from ..api.services import LocalServices

logger = logging.getLogger(__name__)
POLL_SECONDS = 0.5
LEASE_SECONDS = 60
HEARTBEAT_SECONDS = 10
MAX_TASK_COMMANDS = 80
MAX_ARGUMENT_CORRECTIONS = 2
# Older finished tasks are folded into one summary line once the thread is long.
MAX_THREAD_MESSAGES = 900
MAX_EARLIER_TASKS = 20
NEW_STEERING = "New steering arrived for this task"


def open_task(transcript: list[TranscriptEntry]) -> tuple[str, int] | None:
    """The task ID and start index of an unfinished task block, if any."""
    for index in range(len(transcript) - 1, -1, -1):
        item = transcript[index]
        if isinstance(item, TaskEnd):
            return None
        if isinstance(item, TaskStart):
            return str(item.task_id), index
    return None


def pending_command(transcript: list[TranscriptEntry]) -> bool:
    return bool(transcript) and isinstance(transcript[-1], RunCall)


def characters(transcript: list[TranscriptEntry]) -> int:
    return sum(entry_characters(item) for item in transcript)


def fold_finished_tasks(transcript: list[TranscriptEntry]) -> list[TranscriptEntry]:
    """Fold every completed/failed/stopped task block into TaskHistory summaries.

    Keeps any open task verbatim while preventing finished tasks from leaving
    thousands of characters of stale tool outputs in the prompt prefix.
    """
    earlier: list[TaskOutcome] = []
    items = list(transcript)
    if items and isinstance(items[0], TaskHistory):
        earlier = list(items.pop(0).tasks)
    changed = False
    while items:
        end = next(
            (i for i, entry in enumerate(items) if isinstance(entry, TaskEnd)), None
        )
        if end is None:
            break
        finished = items[end]
        earlier = [
            *earlier,
            TaskOutcome(
                task_id=finished.task_id,
                status=finished.status,
                summary=finished.summary[:300],
            ),
        ][-MAX_EARLIER_TASKS:]
        items = items[end + 1 :]
        changed = True
    if not changed:
        return transcript
    return [TaskHistory(tasks=earlier), *items]


def compact(transcript: list[TranscriptEntry], budget: int) -> list[TranscriptEntry]:
    """Drop the oldest finished tasks until the thread fits, keeping their summaries."""
    earlier: list[TaskOutcome] = []
    items = list(transcript)
    if items and isinstance(items[0], TaskHistory):
        earlier = items.pop(0).tasks
    if len(transcript) <= MAX_THREAD_MESSAGES and characters(transcript) <= budget:
        return transcript
    changed = False
    while items:
        end = next(
            (i for i, entry in enumerate(items) if isinstance(entry, TaskEnd)), None
        )
        if end is None:
            break
        finished = items[end]
        earlier = [
            *earlier,
            TaskOutcome(
                task_id=finished.task_id,
                status=finished.status,
                summary=finished.summary[:300],
            ),
        ][-MAX_EARLIER_TASKS:]
        items = items[end + 1 :]
        changed = True
        candidate = [TaskHistory(tasks=earlier), *items]
        if len(candidate) <= MAX_THREAD_MESSAGES and characters(candidate) <= budget:
            return candidate
    return [TaskHistory(tasks=earlier), *items] if changed else transcript


class LeaseLost(Exception):
    """Another worker took this thread; stop without touching it again."""


class ThreadWork:
    """One worker's claim on one agent's computer thread."""

    def __init__(self, tasks: ComputerTasks, thread: ComputerThread):
        self.tasks = tasks
        self.services = tasks.services
        self.key = thread.agent_key
        self.thread = thread
        self.lock = asyncio.Lock()
        self.released = False

    async def save(
        self,
        transcript: list[TranscriptEntry] | None = None,
        consumed: list | None = None,
        task: dict | None = None,
        release: bool = False,
    ) -> None:
        async with self.lock:
            if self.released:
                return
            try:
                value = await self.services.json_request(
                    "Tsuchi",
                    "PUT",
                    f"/api/agents/{self.key}/computer-thread",
                    json={
                        "transcript_format": 2,
                        "transcript": [
                            item.model_dump(mode="json")
                            for item in (
                                self.thread.transcript
                                if transcript is None
                                else transcript
                            )
                        ],
                        "owner_id": None if release else str(self.tasks.owner),
                        "lease_expires_at": None
                        if release
                        else (
                            datetime.now(UTC) + timedelta(seconds=LEASE_SECONDS)
                        ).isoformat(),
                        "expected_version": self.thread.version,
                        "worker_id": str(self.tasks.owner),
                        "consumed_steering_ids": [str(item) for item in consumed or []],
                        "task": task,
                    },
                )
            except ServiceError as error:
                if error.status == 409 and str(error) != NEW_STEERING:
                    raise LeaseLost(str(error)) from error
                raise
            self.thread = ComputerThread.model_validate(value)
            self.released = release

    async def renew(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_SECONDS)
            await self.save()

    async def refresh(self) -> ComputerThread:
        # Held against the heartbeat, whose saves also change the version.
        async with self.lock:
            thread = await self.tasks.thread(self.key)
            if thread.version != self.thread.version:
                raise LeaseLost("The computer thread changed under this worker.")
            self.thread = thread
            return thread

    async def work(self) -> None:
        while True:
            active = await self.tasks.active(self.key)
            if not active:
                await self.save(release=True)
                return
            await self.run_task(active[0])

    def close_interrupted(
        self, transcript: list[TranscriptEntry]
    ) -> list[TranscriptEntry]:
        """End a block left open by a task cancelled while no worker held it."""
        opened = open_task(transcript)
        if opened is None:
            return transcript
        if pending_command(transcript):
            transcript = [
                *transcript,
                RunResult(
                    call_id=transcript[-1].call_id,
                    data={"error": "The task ended before this ran."},
                ),
            ]
        return [
            *transcript,
            TaskEnd(task_id=opened[0], status="stopped", summary=""),
        ]

    @traced("kadan.computer_tasks.run_task")
    async def run_task(self, task: ComputerTask) -> None:
        opened = open_task(self.thread.transcript)
        if task.status == "running":
            # An unrecognized old journal may contain an unconfirmed change.
            # Only the current task needs to stop. Earlier closed tasks are history.
            start = (
                opened[1]
                if opened
                else next(
                    (
                        i + 1
                        for i in reversed(range(len(self.thread.transcript)))
                        if isinstance(self.thread.transcript[i], TaskEnd)
                    ),
                    0,
                )
            )
            if any(
                isinstance(entry, UnrecoveredRecord)
                for entry in self.thread.transcript[start:]
            ):
                await self.end(
                    task,
                    "failed",
                    "Kadan could not safely resume this computer task. Check what it completed before trying again.",
                )
                return
        if opened is None or opened[0] != str(task.id):
            await self.save(
                [
                    *fold_finished_tasks(self.close_interrupted(self.thread.transcript)),
                    TaskStart(
                        task_id=task.id,
                        origin="agent" if task.origin else "user",
                        instructions=task.instructions,
                    ),
                ],
                task={"id": str(task.id), "status": "running"},
            )
        elif task.status == "queued":
            await self.save(task={"id": str(task.id), "status": "running"})
        while True:
            current = await self.tasks.task(self.key, task.id)
            thread = await self.refresh()
            if current.status == "cancelled":
                await self.end(task, "cancelled", "Cancelled by the user.", False)
                return
            steering = [item for item in thread.steering if item.task_id == task.id]
            if steering and not pending_command(thread.transcript):
                await self.save(
                    [
                        *thread.transcript,
                        *(
                            RunMessage(role="user", content=item.content)
                            for item in steering
                        ),
                    ],
                    consumed=[item.id for item in steering],
                )
            replayed = pending_command(self.thread.transcript)
            if replayed:
                command = self.thread.transcript[-1].command
            else:
                try:
                    command = await self.infer()
                except ServiceError as error:
                    await self.end(task, "failed", str(error)[:2000])
                    return
                if command is None:
                    continue
            current = await self.tasks.task(self.key, task.id)
            if current.status == "cancelled":
                await self.refresh()
                await self.end(task, "cancelled", "Cancelled by the user.", False)
                return
            if command.tool in {"finish_task", "finish"}:
                if await self.finish(task, command):
                    return
                continue
            tool = TOOLS_BY_NAME[command.tool]
            tool_started = monotonic()
            result = {
                "tool": tool.name,
                **await invoke_computer_tool(
                    tool, command.arguments, self.services.sandbox, replayed
                ),
            }
            logger.info(
                "Computer tool ended agent=%s task=%s tool=%s elapsed_seconds=%.3f",
                self.key,
                task.id,
                tool.name,
                monotonic() - tool_started,
            )
            await self.save(
                [
                    *self.thread.transcript,
                    RunResult(call_id=self.thread.transcript[-1].call_id, data=result),
                ]
            )

    async def infer(self) -> ToolCommand | None:
        """Choose and persist the next command, or save a rejection and return None."""
        transcript = self.thread.transcript
        opened = open_task(transcript)
        assert opened is not None
        current = transcript[opened[1] :]
        commands = sum(item.role == "call" for item in current)
        rejections = [item for item in current if item.role == "rejection"]
        if len(rejections) > MAX_ARGUMENT_CORRECTIONS:
            raise ServiceError(
                "The computer agent exhausted its tool-argument corrections.", 502
            )
        if commands + len(rejections) >= MAX_TASK_COMMANDS:
            raise ServiceError(
                f"The computer agent reached its {MAX_TASK_COMMANDS}-command task limit.",
                422,
            )
        status = await self.services.inference_status()
        if status.state != "ready":
            raise ServiceError(status.message, 503)
        system = RunMessage(role="system", content=await self.tasks.instructions())
        budget = (
            context_characters(getattr(status, "context_limit", None))
            - tool_characters(COMPUTER_AGENT_TOOLS)
            - len(system.content)
        )
        transcript = compact(fold_finished_tasks(transcript), budget)
        if characters(transcript) > budget:
            raise ServiceError(
                "The task exceeded the computer agent's context limit.", 422
            )
        started = monotonic()
        logger.info(
            "Computer inference started agent=%s task=%s chars=%d",
            self.key,
            opened[0],
            characters(transcript) + len(system.content),
        )
        try:
            call = await self.tasks.inference.call(
                transcript_messages([system, *transcript]),
                status.model_id,
                f"kadan-computer:{self.key}",
                COMPUTER_AGENT_TOOLS,
                priority="background",
            )
        finally:
            logger.info(
                "Computer inference ended agent=%s task=%s elapsed_seconds=%.3f",
                self.key,
                opened[0],
                monotonic() - started,
            )
        if call.command.tool == "finish":
            call = RunCall(
                call_id=call.call_id,
                command=ToolCommand(
                    tool="finish_task",
                    arguments={"summary": "Stopped without reporting a result."},
                ),
            )
        command = call.command
        rejection = validate(command, COMPUTER_AGENT_TOOLS)
        if rejection is not None:
            await self.save(
                [
                    *transcript,
                    RunObservation(
                        role="rejection",
                        name=command.tool,
                        data=rejection.model_dump(mode="json"),
                    ),
                ]
            )
            return None
        # Persist the action before side effects, as conversation runs do.
        await self.save(
            [
                *transcript,
                call,
            ]
        )
        return command

    async def finish(self, task: ComputerTask, command: ToolCommand) -> bool:
        """End the task; False when new steering arrived first."""
        args = FinishTaskArguments.model_validate(command.arguments)
        try:
            await self.end(
                task, "completed" if args.succeeded else "failed", args.summary
            )
        except ServiceError as error:
            if str(error) != NEW_STEERING:
                raise
            result = {
                "tool": "finish_task",
                "error": "The user sent new steering before this task finished. "
                "Read it and continue.",
            }
            await self.save(
                [
                    *self.thread.transcript,
                    RunResult(call_id=self.thread.transcript[-1].call_id, data=result),
                ]
            )
            return False
        return True

    async def end(
        self, task: ComputerTask, status: str, summary: str, update: bool = True
    ) -> None:
        transcript = list(self.thread.transcript)
        if pending_command(transcript):
            result = (
                {"tool": "finish_task", "finished": True}
                if status != "cancelled"
                else {"error": "The user cancelled this task before it ran."}
            )
            transcript.append(
                RunResult(call_id=self.thread.transcript[-1].call_id, data=result)
            )
        transcript.append(TaskEnd(task_id=task.id, status=status, summary=summary))
        while True:
            try:
                await self.save(
                    transcript,
                    # Steering for a task that did not succeed ends with it; a
                    # completed task must read it first.
                    consumed=[
                        item.id
                        for item in self.thread.steering
                        if item.task_id == task.id
                    ]
                    if status != "completed"
                    else None,
                    task={"id": str(task.id), "status": status, "summary": summary}
                    if update
                    else None,
                )
                return
            except ServiceError as error:
                if str(error) != NEW_STEERING or status == "completed":
                    raise
                await self.refresh()


class ComputerTasks:
    def __init__(self, services: LocalServices):
        self.services = services
        self.owner = uuid4()
        self.wakeup = asyncio.Event()
        self.inference = AgentInference(services.client, services.settings.hikari_url)
        self.work: dict[str, asyncio.Task] = {}
        self.runner: asyncio.Task | None = None

    def start(self) -> None:
        if self.runner is None:
            self.runner = asyncio.create_task(self.run(), name="computer-tasks")

    async def close(self) -> None:
        tasks = [*self.work.values(), *([self.runner] if self.runner else [])]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.work.clear()
        self.runner = None

    async def run(self) -> None:
        while True:
            if self.services.sandbox.configured:
                try:
                    await self.dispatch()
                except ServiceError as error:
                    logger.warning("Computer tasks could not be read: %s", error)
            try:
                await asyncio.wait_for(self.wakeup.wait(), timeout=POLL_SECONDS)
            except TimeoutError:
                pass
            self.wakeup.clear()

    async def instructions(self) -> str:
        path = Path(self.services.settings.computer_instruction_file)
        try:
            return await asyncio.to_thread(path.read_text)
        except (OSError, UnicodeError) as error:
            raise ServiceError(
                f"The computer agent instructions file {path} could not be read."
            ) from error

    async def active(self, key: str | None = None) -> list[ComputerTask]:
        path = f"/api/agents/{key}/computer-tasks" if key else "/api/computer-tasks"
        return [
            ComputerTask.model_validate(value)
            for value in await self.services.json_request("Tsuchi", "GET", path)
        ]

    async def task(self, key: str, task_id) -> ComputerTask:
        return ComputerTask.model_validate(
            await self.services.json_request(
                "Tsuchi", "GET", f"/api/agents/{key}/computer-tasks/{task_id}"
            )
        )

    async def thread(self, key: str) -> ComputerThread:
        return ComputerThread.model_validate(
            await self.services.json_request(
                "Tsuchi", "GET", f"/api/agents/{key}/computer-thread"
            )
        )

    async def dispatch(self) -> None:
        for key, work in list(self.work.items()):
            if work.done():
                result = (await asyncio.gather(work, return_exceptions=True))[0]
                if isinstance(result, Exception):
                    logger.warning("Computer work for %s ended: %s", key, result)
                del self.work[key]
        for key in dict.fromkeys(task.agent_key for task in await self.active()):
            if key not in self.work:
                self.work[key] = asyncio.create_task(
                    self.serve(key), name=f"computer-{key}"
                )

    async def serve(self, key: str) -> None:
        work = ThreadWork(self, await self.thread(key))
        try:
            await work.save()
        except LeaseLost:
            return
        heartbeat = asyncio.create_task(work.renew())
        tasks = asyncio.create_task(work.work())
        try:
            done, _ = await asyncio.wait(
                {tasks, heartbeat}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()
        except LeaseLost as error:
            logger.info("Computer thread for %s was taken over: %s", key, error)
        finally:
            tasks.cancel()
            heartbeat.cancel()
            await asyncio.gather(tasks, heartbeat, return_exceptions=True)
