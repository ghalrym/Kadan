"""First-class, replay-safe native Work delegation tools for Kadan chat agents.

Every mutating launch/send call passes a deterministic `client_request_id` derived from
`conversation.action_id` so crash-recovery replays never create duplicate threads or turns.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, JsonValue

from ...api.config import ServiceError
from ...api.pydantic_models.work import (
    WorkBucket,
    WorkPendingRespondRequest,
    WorkRunCreate,
    WorkThreadCreate,
    WorkThreadOrganizeRequest,
    WorkWorkspaceMode,
)
from ...work.client import work_thread_markdown_link, work_thread_url
from .base import Arguments, NoArguments, conversation_tool


class WorkThreadListArguments(Arguments):
    bucket: WorkBucket | None = Field(
        default=None,
        description="Filter by bucket: default, working, snoozed, or settled.",
    )
    project_id: UUID | None = Field(
        default=None,
        description="Optional project ID to filter threads by.",
    )
    query: str = Field(
        default="",
        max_length=200,
        description="Optional text filter matching thread title or branch.",
    )
    limit: int = Field(default=30, ge=1, le=100)


class WorkThreadReadArguments(Arguments):
    thread_id: UUID = Field(description="ID of the native Work thread to read.")
    activity_limit: int = Field(
        default=40,
        ge=1,
        le=200,
        description="Maximum number of recent activity items to return.",
    )


class WorkThreadLaunchArguments(Arguments):
    project_id: UUID = Field(
        description="Registered Work project ID from work_project_list."
    )
    title: str = Field(
        min_length=1,
        max_length=200,
        description="Short descriptive title for the coding task.",
    )
    prompt: str = Field(
        min_length=1,
        max_length=100000,
        description="Clear, self-contained task prompt for the coding harness.",
    )
    harness: str = Field(
        default="codex",
        pattern=r"^[a-z0-9][a-z0-9_-]{0,47}$",
        description="Coding harness to use (codex, claude, antigravity, grok).",
    )
    model: str | None = Field(
        default=None,
        max_length=256,
        description="Optional model ID supported by the chosen harness.",
    )
    workspace_mode: WorkWorkspaceMode = Field(
        default="worktree",
        description="Use 'worktree' for an isolated branch copy or 'root' for the project folder.",
    )
    base_ref: str | None = Field(
        default=None,
        max_length=256,
        description="Optional Git base branch or ref when using worktree mode.",
    )
    branch: str | None = Field(
        default=None,
        max_length=256,
        description="Optional Git branch name when using worktree mode.",
    )


class WorkThreadSendArguments(Arguments):
    thread_id: UUID = Field(description="ID of the native Work thread to message.")
    prompt: str = Field(
        min_length=1,
        max_length=100000,
        description="Follow-up instruction or steering update.",
    )
    mode: Literal["send", "steer", "queue"] = Field(
        default="send",
        description="'send' starts or queues, 'steer' updates an active turn inline when supported, 'queue' queues after the current turn.",
    )


class WorkThreadInterruptArguments(Arguments):
    thread_id: UUID = Field(description="ID of the native Work thread to stop.")


class WorkPendingRequestArguments(Arguments):
    thread_id: UUID = Field(description="ID of the native Work thread.")
    action: Literal["list", "read", "respond"] = Field(
        description="List pending questions/approvals, read one, or submit a response."
    )
    request_id: str | None = Field(
        default=None,
        max_length=256,
        description="Required for 'read' and 'respond'.",
    )
    response: dict[str, JsonValue] | None = Field(
        default=None,
        description="Answer payload for 'respond' (e.g. {'answer': '...'} or {'decision': 'approve'|'deny'}).",
    )


class WorkThreadOrganizeArguments(Arguments):
    thread_id: UUID = Field(description="ID of the native Work thread.")
    action: Literal["default", "snooze", "settle", "unsettle"] = Field(
        description="Move thread to default, snooze it, settle it, or unsettle it."
    )
    snooze_minutes: int | None = Field(
        default=None,
        ge=1,
        le=43200,
        description="Minutes to snooze when action is 'snooze'.",
    )
    snoozed_until: datetime | None = Field(
        default=None,
        description="ISO timestamp with timezone when action is 'snooze'.",
    )


def _work_client(conversation):
    return getattr(conversation.runs.services, "work", None)


@conversation_tool
async def work_project_list(args: NoArguments, conversation) -> dict:
    """List registered native Work projects and available coding harnesses."""
    work = _work_client(conversation)
    if work is None:
        return {"error": "Native Work is not available in this environment."}
    try:
        projects = await work.list_projects()
        harnesses = await work.list_harnesses()
    except ServiceError as exc:
        return {"error": str(exc)}
    return {
        "projects": [
            {
                "id": str(p.id),
                "title": p.title,
                "default_branch": p.default_branch,
            }
            for p in projects
        ],
        "harnesses": [
            {
                "id": h.id,
                "display_name": h.display_name,
                "installed": h.installed,
                "auth_status": h.auth.status,
                "default_model": h.default_model,
            }
            for h in harnesses
        ],
    }


@conversation_tool
async def work_thread_list(args: WorkThreadListArguments, conversation) -> dict:
    """List native Work threads, optionally filtered by bucket, project, or search text."""
    work = _work_client(conversation)
    if work is None:
        return {"error": "Native Work is not available in this environment."}
    try:
        threads = await work.list_threads(
            bucket=args.bucket,
            project_id=args.project_id,
            q=args.query,
            limit=args.limit,
        )
    except ServiceError as exc:
        return {"error": str(exc)}
    return {
        "threads": [
            {
                "id": str(t.id),
                "title": t.title,
                "project_id": str(t.project_id),
                "harness": t.harness,
                "model": t.model,
                "bucket": t.bucket,
                "run_status": t.run_status,
                "branch": t.branch,
                "pending_count": sum(
                    1 for p in t.pending_requests if p.status == "pending"
                ),
                "url": work_thread_url(t.id),
                "link": work_thread_markdown_link(t.id, t.title),
            }
            for t in threads
        ]
    }


@conversation_tool
async def work_thread_read(args: WorkThreadReadArguments, conversation) -> dict:
    """Read a native Work thread's status, pending requests, recent runs, and activity."""
    work = _work_client(conversation)
    if work is None:
        return {"error": "Native Work is not available in this environment."}
    try:
        detail = await work.get_thread_detail(
            args.thread_id, activity_limit=args.activity_limit
        )
    except ServiceError as exc:
        return {"error": str(exc)}
    thread = detail.thread
    return {
        "thread": {
            "id": str(thread.id),
            "title": thread.title,
            "project": detail.project.title if detail.project else None,
            "harness": thread.harness,
            "model": thread.model,
            "workspace_mode": thread.workspace_mode,
            "branch": thread.branch,
            "bucket": thread.bucket,
            "run_status": thread.run_status,
            "snoozed_until": (
                thread.snoozed_until.isoformat() if thread.snoozed_until else None
            ),
            "error": thread.error,
            "url": work_thread_url(thread.id),
            "link": work_thread_markdown_link(thread.id, thread.title),
        },
        "pending_requests": [
            {
                "id": req.id,
                "kind": req.kind,
                "title": req.title,
                "prompt": req.prompt,
                "options": req.options,
                "tool_name": req.tool_name,
                "status": req.status,
            }
            for req in thread.pending_requests
            if req.status == "pending"
        ],
        "activity": [
            {
                "position": item.position,
                "kind": item.kind,
                "text": item.text[:2000],
                "data": item.data,
            }
            for item in detail.activity[-args.activity_limit :]
        ],
    }


@conversation_tool
async def work_thread_launch(args: WorkThreadLaunchArguments, conversation) -> dict:
    """Launch a new native Work coding thread and start its first turn.

    Replay-safe via deterministic action_id. Kadan automatically wakes this chat
    conversation when the thread pauses for a question/approval or finishes.
    """
    work = _work_client(conversation)
    if work is None:
        return {"error": "Native Work is not available in this environment."}
    try:
        thread, run = await work.create_thread(
            WorkThreadCreate(
                project_id=args.project_id,
                title=args.title,
                harness=args.harness,
                model=args.model,
                workspace_mode=args.workspace_mode,
                base_ref=args.base_ref,
                branch=args.branch,
                prompt=args.prompt,
                client_request_id=f"work-launch:{conversation.action_id}",
            ),
            origin_agent_key=conversation.run.agent_key,
            origin_conversation_id=conversation.run.conversation_id,
            origin_run_id=conversation.run.id,
        )
    except ServiceError as exc:
        return {"error": str(exc)}
    return {
        "thread_id": str(thread.id),
        "run_id": str(run.id) if run is not None else None,
        "title": thread.title,
        "harness": thread.harness,
        "model": thread.model,
        "branch": thread.branch,
        "url": work_thread_url(thread.id),
        "link": work_thread_markdown_link(thread.id, thread.title),
        "note": (
            "Kadan will wake you automatically in this chat when this Work thread "
            "finishes or pauses for a question or approval. Share the link with the user "
            "and end this turn without polling."
        ),
    }


@conversation_tool
async def work_thread_send(args: WorkThreadSendArguments, conversation) -> dict:
    """Send a follow-up message, steering update, or queued turn to a native Work thread."""
    work = _work_client(conversation)
    if work is None:
        return {"error": "Native Work is not available in this environment."}
    try:
        thread = await work.get_thread(args.thread_id)
        run = await work.send_message(
            args.thread_id,
            WorkRunCreate(
                prompt=args.prompt,
                mode=args.mode,
                client_request_id=f"work-send:{conversation.action_id}",
            ),
        )
    except ServiceError as exc:
        return {"error": str(exc)}
    return {
        "thread_id": str(thread.id),
        "run_id": str(run.id),
        "mode": run.mode,
        "status": run.status,
        "url": work_thread_url(thread.id),
        "link": work_thread_markdown_link(thread.id, thread.title),
    }


@conversation_tool
async def work_thread_interrupt(
    args: WorkThreadInterruptArguments, conversation
) -> dict:
    """Interrupt or cancel active work on a native Work thread."""
    work = _work_client(conversation)
    if work is None:
        return {"error": "Native Work is not available in this environment."}
    try:
        thread = await work.interrupt_thread(args.thread_id)
    except ServiceError as exc:
        return {"error": str(exc)}
    return {
        "thread_id": str(thread.id),
        "run_status": thread.run_status,
        "bucket": thread.bucket,
        "url": work_thread_url(thread.id),
        "link": work_thread_markdown_link(thread.id, thread.title),
    }


@conversation_tool
async def work_pending_request(args: WorkPendingRequestArguments, conversation) -> dict:
    """List, read, or answer a pending question or approval on a native Work thread."""
    work = _work_client(conversation)
    if work is None:
        return {"error": "Native Work is not available in this environment."}
    try:
        thread = await work.get_thread(args.thread_id)
        if args.action == "list":
            return {
                "thread_id": str(thread.id),
                "pending_requests": [
                    req.model_dump(mode="json")
                    for req in thread.pending_requests
                    if req.status == "pending"
                ],
            }
        if not args.request_id:
            return {"error": "request_id is required for read and respond actions."}
        if args.action == "read":
            req = next(
                (r for r in thread.pending_requests if r.id == args.request_id), None
            )
            if req is None:
                return {"error": "Pending request not found on this thread."}
            return req.model_dump(mode="json")
        if args.response is None:
            return {"error": "response payload is required when action is 'respond'."}
        updated = await work.respond_pending_request(
            args.thread_id,
            args.request_id,
            WorkPendingRespondRequest(response=args.response),
        )
        return {
            "thread_id": str(updated.id),
            "request_id": args.request_id,
            "run_status": updated.run_status,
            "bucket": updated.bucket,
            "url": work_thread_url(updated.id),
            "link": work_thread_markdown_link(updated.id, updated.title),
        }
    except ServiceError as exc:
        return {"error": str(exc)}


@conversation_tool
async def work_thread_organize(args: WorkThreadOrganizeArguments, conversation) -> dict:
    """Move a native Work thread between Default, Snoozed, and Settled buckets."""
    work = _work_client(conversation)
    if work is None:
        return {"error": "Native Work is not available in this environment."}
    try:
        updated = await work.organize_thread(
            args.thread_id,
            WorkThreadOrganizeRequest(
                action=args.action,
                snooze_minutes=args.snooze_minutes,
                snoozed_until=args.snoozed_until,
            ),
        )
    except ServiceError as exc:
        return {"error": str(exc)}
    return {
        "thread_id": str(updated.id),
        "bucket": updated.bucket,
        "snoozed_until": (
            updated.snoozed_until.isoformat() if updated.snoozed_until else None
        ),
        "url": work_thread_url(updated.id),
        "link": work_thread_markdown_link(updated.id, updated.title),
    }


TOOLS = [
    work_project_list,
    work_thread_list,
    work_thread_read,
    work_thread_launch,
    work_thread_send,
    work_thread_interrupt,
    work_pending_request,
    work_thread_organize,
]
