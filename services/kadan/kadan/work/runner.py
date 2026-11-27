"""Background Work execution & snooze reconciliation service inside `kadan-runner`.

Responsibilities:
1. Wake expired snoozed threads back to `default` (clearing `snoozed_until`).
2. Claim durable `WorkRun`s from Tsuchi, execute them directly in `project.root_path`
   via the configured `HarnessAdapter`, stream normalized activity into Tsuchi, renew
   leases with a background heartbeat during long silent turns, and handle live steering,
   cancellation, and pending user questions & approvals.
3. Emit idempotent Tsuchi `event-run` wakeups (`kind="work"`) directly into originating
   Kadan chat conversations when an agent-launched Work thread pauses for input or enters
   a terminal state (event-driven, with zero chat polling).
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4, uuid5

from ..api.config import ServiceError
from ..api.pydantic_models.work import (
    WorkProject,
    WorkRun,
    WorkThread,
)
from .client import work_thread_markdown_link, work_thread_url
from .contract import (
    HarnessError,
    HarnessSession,
    SessionConfig,
)

if TYPE_CHECKING:
    from ..api.services import LocalServices

logger = logging.getLogger(__name__)

RECONCILE_SECONDS = 1.0
LEASE_SECONDS = 90
LEASE_HEARTBEAT_SECONDS = 30.0
PENDING_POLL_SECONDS = 0.5
DELTA_FLUSH_CHARS = 240
MAX_WAKEUP_SUMMARY_CHARS = 8000


class _ActiveExecution:
    def __init__(self, run: WorkRun, thread_id: UUID):
        self.run = run
        self.thread_id = thread_id
        self.session: HarnessSession | None = None
        self.task: asyncio.Task[None] | None = None
        self.run_lock = asyncio.Lock()
        self.cancelled = False


class WorkRunner:
    """Executes durable native Work runs and reconciles snoozed threads."""

    def __init__(self, services: LocalServices, *, owner_id: UUID | None = None):
        self.services = services
        self.owner_id = owner_id or uuid4()
        self._loop_task: asyncio.Task[None] | None = None
        self._active: dict[UUID, _ActiveExecution] = {}  # keyed by thread_id

    def start(self) -> None:
        if self._loop_task is None:
            self._loop_task = asyncio.create_task(self.run(), name="kadan-work-runner")

    async def close(self) -> None:
        if self._loop_task is not None:
            self._loop_task.cancel()
            await asyncio.gather(self._loop_task, return_exceptions=True)
            self._loop_task = None
        for active in list(self._active.values()):
            active.cancelled = True
            if active.session is not None:
                with suppress(Exception):
                    await active.session.interrupt()
            if active.task is not None and not active.task.done():
                active.task.cancel()
        if self._active:
            await asyncio.gather(
                *(a.task for a in self._active.values() if a.task is not None),
                return_exceptions=True,
            )
            self._active.clear()

    async def run(self) -> None:
        while True:
            try:
                await self.reconcile_once()
            except asyncio.CancelledError:
                raise
            except ServiceError as error:
                logger.warning("WorkRunner reconcile failed: %s", error)
            except Exception:
                logger.exception("Unexpected error in WorkRunner reconcile")
            await asyncio.sleep(RECONCILE_SECONDS)

    async def reconcile_once(self) -> None:
        # 1. Reconcile expired snoozes back into `default`.
        with suppress(ServiceError):
            await self.services.work.wake_snoozed()

        # 2. Fetch active work runs from Tsuchi.
        raw_runs = await self.services.json_request(
            "Tsuchi", "GET", "/api/work/runs", params={"active_only": "true"}
        )
        runs = [WorkRun.model_validate(item) for item in raw_runs]
        runs_by_id = {r.id: r for r in runs}

        # 3. Clean up completed tasks and cancel runs cancelled externally.
        for thread_id, active in list(self._active.items()):
            current = runs_by_id.get(active.run.id)
            if current is None or current.status in {
                "completed",
                "failed",
                "interrupted",
                "cancelled",
            }:
                active.cancelled = True
                if active.session is not None:
                    with suppress(Exception):
                        await active.session.interrupt()
                if active.task is not None and not active.task.done():
                    active.task.cancel()
            if active.task is not None and active.task.done():
                with suppress(Exception):
                    active.task.result()
                self._active.pop(thread_id, None)

        # 4. Deliver live steering runs to active sessions if supported.
        # Runs carrying image attachments are never consumed by text-only steer;
        # they stay queued and run as the next full turn with images intact.
        for run in sorted(runs, key=lambda item: (item.ordinal, item.created_at)):
            active = self._active.get(run.thread_id)
            if (
                active is not None
                and run.id != active.run.id
                and run.status == "queued"
                and run.mode == "steer"
                and not run.images
                and active.session is not None
            ):
                steered = False
                with suppress(Exception):
                    steered = await active.session.steer(run.prompt)
                if steered:
                    await self._append_activity(
                        run.thread_id,
                        [
                            {
                                "id": f"run-{run.id}-user",
                                "run_id": str(run.id),
                                "kind": "user_message",
                                "text": run.prompt,
                                "data": {"mode": "steer"},
                            }
                        ],
                    )
                    with suppress(ServiceError):
                        await self.services.json_request(
                            "Tsuchi",
                            "PUT",
                            f"/api/work/runs/{run.id}",
                            json={
                                "status": "completed",
                                "owner_id": None,
                                "lease_expires_at": None,
                                "final_message": "Delivered live steering update.",
                                "expected_version": run.version,
                            },
                        )

        # 5. Claim next runnable turn per idle thread.
        now = datetime.now(UTC)
        for run in sorted(runs, key=lambda item: (item.ordinal, item.created_at)):
            if run.thread_id in self._active:
                continue
            claimable = run.status == "queued" or (
                run.status in {"preparing", "running", "waiting"}
                and (run.lease_expires_at is None or run.lease_expires_at <= now)
            )
            if not claimable:
                continue
            try:
                claimed = WorkRun.model_validate(
                    await self.services.json_request(
                        "Tsuchi",
                        "PUT",
                        f"/api/work/runs/{run.id}",
                        json={
                            "status": "preparing",
                            "owner_id": str(self.owner_id),
                            "lease_expires_at": (
                                now + timedelta(seconds=LEASE_SECONDS)
                            ).isoformat(),
                            "expected_version": run.version,
                        },
                    )
                )
            except ServiceError:
                continue
            active = _ActiveExecution(claimed, claimed.thread_id)
            active.task = asyncio.create_task(
                self._execute_run(active),
                name=f"work-run-{claimed.id}",
            )
            self._active[claimed.thread_id] = active

    async def _update_run(
        self,
        active: _ActiveExecution,
        *,
        status: str,
        keep_lease: bool,
        error: str | None = None,
        final_message: str | None = None,
        wakeup_delivered: bool | None = None,
    ) -> WorkRun:
        async with active.run_lock:
            now = datetime.now(UTC)
            payload: dict[str, Any] = {
                "status": status,
                "owner_id": str(self.owner_id) if keep_lease else None,
                "lease_expires_at": (
                    (now + timedelta(seconds=LEASE_SECONDS)).isoformat()
                    if keep_lease
                    else None
                ),
                "error": error,
                "final_message": final_message,
                "expected_version": active.run.version,
            }
            if wakeup_delivered is not None:
                payload["wakeup_delivered"] = wakeup_delivered
            updated = WorkRun.model_validate(
                await self.services.json_request(
                    "Tsuchi",
                    "PUT",
                    f"/api/work/runs/{active.run.id}",
                    json=payload,
                )
            )
            active.run = updated
            return updated

    async def _lease_heartbeat(self, active: _ActiveExecution) -> None:
        """Renew active run lease periodically even while a harness is silent."""
        try:
            while not active.cancelled:
                await asyncio.sleep(LEASE_HEARTBEAT_SECONDS)
                if active.cancelled:
                    return
                current_status = active.run.status
                if current_status not in {"preparing", "running", "waiting"}:
                    return
                with suppress(ServiceError):
                    await self._update_run(
                        active, status=current_status, keep_lease=True
                    )
        except asyncio.CancelledError:
            return

    async def _append_activity(
        self, thread_id: UUID, items: list[dict[str, Any]]
    ) -> None:
        if not items:
            return
        with suppress(ServiceError):
            await self.services.json_request(
                "Tsuchi",
                "POST",
                f"/api/work/threads/{thread_id}/activity",
                json=items,
            )

    async def _emit_chat_wakeup(
        self,
        thread: WorkThread,
        *,
        event_id: UUID,
        payload: dict[str, Any],
    ) -> bool:
        if (
            not thread.origin_agent_key
            or thread.origin_conversation_id is None
            or thread.origin_run_id is None
        ):
            return False
        enriched = {
            **payload,
            "thread_id": str(thread.id),
            "thread_title": thread.title,
            "thread_url": work_thread_url(thread.id),
            "thread_link": work_thread_markdown_link(thread.id, thread.title),
            "harness": thread.harness,
        }
        try:
            await self.services.json_request(
                "Tsuchi",
                "POST",
                f"/api/agents/{thread.origin_agent_key}/resources/"
                f"{thread.origin_conversation_id}/event-runs",
                json={
                    "id": str(event_id),
                    "origin_run_id": str(thread.origin_run_id),
                    "event": {
                        "kind": "work",
                        "content": json.dumps(enriched, ensure_ascii=False),
                    },
                },
            )
            return True
        except ServiceError as error:
            if error.status in {404, 409}:
                logger.info(
                    "Work chat wakeup for thread %s dropped: %s", thread.id, error
                )
                return False
            logger.warning(
                "Could not deliver Work chat wakeup for thread %s: %s",
                thread.id,
                error,
            )
            return False

    async def _wait_for_pending_response(
        self, active: _ActiveExecution, request_id: str
    ) -> dict[str, Any] | None:
        while not active.cancelled:
            try:
                latest_run = WorkRun.model_validate(
                    await self.services.json_request(
                        "Tsuchi", "GET", f"/api/work/runs/{active.run.id}"
                    )
                )
                if latest_run.status in {"cancelled", "interrupted", "failed"}:
                    return None
                thread = await self.services.work.get_thread(active.thread_id)
                for req in thread.pending_requests:
                    if req.id == request_id:
                        if req.status == "answered" and req.response is not None:
                            return dict(req.response)
                        if req.status == "cancelled":
                            return None
            except ServiceError:
                pass
            await asyncio.sleep(PENDING_POLL_SECONDS)
        return None

    async def _execute_run(self, active: _ActiveExecution) -> None:
        run = active.run
        session: HarnessSession | None = None
        thread: WorkThread | None = None
        final_status = "failed"
        final_text = ""
        error_msg: str | None = None
        heartbeat = asyncio.create_task(self._lease_heartbeat(active))
        try:
            thread = await self.services.work.get_thread(run.thread_id)
            project: WorkProject = await self.services.work.get_project(
                thread.project_id
            )

            working_dir = project.root_path
            if not Path(working_dir).is_dir():
                raise HarnessError(
                    "The project folder could not be found on this server."
                )

            # Record user prompt in activity log.
            user_data: dict[str, Any] = {"mode": run.mode}
            if run.images:
                user_data["image_count"] = len(run.images)
            await self._append_activity(
                thread.id,
                [
                    {
                        "id": f"run-{run.id}-user",
                        "run_id": str(run.id),
                        "kind": "user_message",
                        "text": run.prompt,
                        "data": user_data,
                    }
                ],
            )

            run = await self._update_run(active, status="running", keep_lease=True)

            adapter = self.services.work.registry.get(thread.harness)
            priority = (
                "background" if thread.origin_agent_key is not None else "interactive"
            )
            config = SessionConfig(
                thread_id=thread.id,
                working_dir=working_dir,
                model=thread.model,
                model_options=dict(thread.model_options),
                harness_session_id=thread.harness_session_id,
                priority=priority,
            )
            session = await adapter.open_session(config, self.services)
            active.session = session

            seq = 0
            text_buf: list[str] = []
            reasoning_buf: list[str] = []

            async def flush_buffers() -> None:
                nonlocal seq
                items: list[dict[str, Any]] = []
                if reasoning_buf:
                    seq += 1
                    items.append(
                        {
                            "id": f"run-{run.id}-ev-{seq}",
                            "run_id": str(run.id),
                            "kind": "reasoning",
                            "text": "".join(reasoning_buf),
                            "data": {},
                        }
                    )
                    reasoning_buf.clear()
                if text_buf:
                    seq += 1
                    items.append(
                        {
                            "id": f"run-{run.id}-ev-{seq}",
                            "run_id": str(run.id),
                            "kind": "assistant_message",
                            "text": "".join(text_buf),
                            "data": {"streaming": True},
                        }
                    )
                    text_buf.clear()
                if items:
                    await self._append_activity(thread.id, items)

            if run.images and getattr(session, "supports_images", False):
                turn_stream = session.start_turn(run.prompt, images=run.images)  # type: ignore[call-arg]
            else:
                turn_stream = session.start_turn(run.prompt)

            async for event in turn_stream:
                if event.kind == "session_started" and event.session_id:
                    if thread.harness_session_id != event.session_id:
                        with suppress(ServiceError):
                            latest_t = await self.services.work.get_thread(thread.id)
                            thread = WorkThread.model_validate(
                                await self.services.json_request(
                                    "Tsuchi",
                                    "PUT",
                                    f"/api/work/threads/{thread.id}",
                                    json={
                                        "harness_session_id": event.session_id,
                                        "expected_version": latest_t.version,
                                    },
                                )
                            )
                elif event.kind == "text_delta":
                    text_buf.append(event.text)
                    final_text += event.text
                    if sum(len(p) for p in text_buf) >= DELTA_FLUSH_CHARS:
                        await flush_buffers()
                elif event.kind == "reasoning_delta":
                    reasoning_buf.append(event.text)
                    if sum(len(p) for p in reasoning_buf) >= DELTA_FLUSH_CHARS:
                        await flush_buffers()
                elif event.kind == "assistant_message":
                    text_buf.clear()
                    await flush_buffers()
                    if event.text:
                        final_text = event.text
                        seq += 1
                        await self._append_activity(
                            thread.id,
                            [
                                {
                                    "id": f"run-{run.id}-ev-{seq}",
                                    "run_id": str(run.id),
                                    "kind": "assistant_message",
                                    "text": event.text,
                                    "data": {},
                                }
                            ],
                        )
                elif event.kind in {
                    "tool_call",
                    "tool_result",
                    "file_change",
                    "status",
                    "error",
                }:
                    await flush_buffers()
                    seq += 1
                    data: dict[str, Any] = {}
                    if event.call_id:
                        data["call_id"] = event.call_id
                    if event.tool_name:
                        data["tool_name"] = event.tool_name
                    if event.arguments is not None:
                        data["arguments"] = event.arguments
                    if event.file_path:
                        data["file_path"] = event.file_path
                    if event.change_type:
                        data["change_type"] = event.change_type
                    if event.is_error:
                        data["is_error"] = True
                    if event.kind == "status" and event.raw:
                        for diag_key in ("telemetry", "compaction", "guardrail"):
                            if diag_key in event.raw:
                                data[diag_key] = event.raw[diag_key]
                    await self._append_activity(
                        thread.id,
                        [
                            {
                                "id": f"run-{run.id}-ev-{seq}",
                                "run_id": str(run.id),
                                "kind": event.kind,
                                "text": event.text or event.error or "",
                                "data": data,
                            }
                        ],
                    )
                elif event.kind in {"pending_question", "pending_approval"}:
                    await flush_buffers()
                    req_id = event.request_id or f"req-{run.id}-{seq + 1}"
                    pending_kind = (
                        "question" if event.kind == "pending_question" else "approval"
                    )
                    # Emit idempotent chat wakeup before flipping thread to `waiting`
                    # so observers that see `waiting` also observe the queued wakeup.
                    await self._emit_chat_wakeup(
                        thread,
                        event_id=uuid5(run.id, f"work-pending-{req_id}"),
                        payload={
                            "event_type": f"pending_{pending_kind}",
                            "request_id": req_id,
                            "title": event.title,
                            "prompt": event.text,
                            "tool_name": event.tool_name,
                            "options": event.options,
                        },
                    )
                    thread = WorkThread.model_validate(
                        await self.services.json_request(
                            "Tsuchi",
                            "POST",
                            f"/api/work/threads/{thread.id}/pending-requests",
                            json={
                                "id": req_id,
                                "kind": pending_kind,
                                "title": event.title,
                                "prompt": event.text,
                                "options": event.options,
                                "tool_name": event.tool_name,
                                "arguments": event.arguments,
                            },
                        )
                    )
                    seq += 1
                    await self._append_activity(
                        thread.id,
                        [
                            {
                                "id": f"run-{run.id}-pending-{req_id}",
                                "run_id": str(run.id),
                                "kind": "pending_request",
                                "text": event.text,
                                "data": {
                                    "request_id": req_id,
                                    "pending_kind": pending_kind,
                                    "title": event.title,
                                    "tool_name": event.tool_name,
                                },
                            }
                        ],
                    )
                    run = await self._update_run(
                        active, status="waiting", keep_lease=True
                    )

                    answer = await self._wait_for_pending_response(active, req_id)
                    if answer is None:
                        await session.interrupt()
                        final_status = "cancelled"
                        break
                    await session.respond_pending(req_id, answer)
                    run = await self._update_run(
                        active, status="running", keep_lease=True
                    )
                elif event.kind == "turn_completed":
                    await flush_buffers()
                    if event.text and not final_text:
                        final_text = event.text
                    final_status = event.terminal_status or "completed"
                    error_msg = event.error
                    break
            else:
                await flush_buffers()
                if final_status == "failed" and not error_msg:
                    final_status = "interrupted" if active.cancelled else "completed"

        except asyncio.CancelledError:
            final_status = "cancelled"
            if session is not None:
                with suppress(Exception):
                    await session.interrupt()
        except (HarnessError, ServiceError) as exc:
            final_status = "failed"
            error_msg = str(exc)
            if thread is not None:
                await self._append_activity(
                    thread.id,
                    [
                        {
                            "id": f"run-{run.id}-error",
                            "run_id": str(run.id),
                            "kind": "error",
                            "text": error_msg,
                            "data": {},
                        }
                    ],
                )
        except Exception as exc:
            logger.exception("Work run %s failed unexpectedly", run.id)
            final_status = "failed"
            error_msg = str(exc) or "Unexpected harness error."
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

            if session is not None:
                with suppress(Exception):
                    await session.close()
                active.session = None

            # Emit terminal chat wakeup before persisting terminal run status so
            # observers that see `completed`/`failed` also observe the queued wakeup.
            wakeup_delivered = False
            if thread is not None and not active.run.wakeup_delivered:
                wakeup_delivered = await self._emit_chat_wakeup(
                    thread,
                    event_id=uuid5(run.id, "work-done"),
                    payload={
                        "event_type": "thread_settled",
                        "status": final_status,
                        "message": final_text[:MAX_WAKEUP_SUMMARY_CHARS],
                        "error": error_msg,
                    },
                )

            with suppress(ServiceError):
                async with active.run_lock:
                    latest_run = WorkRun.model_validate(
                        await self.services.json_request(
                            "Tsuchi", "GET", f"/api/work/runs/{run.id}"
                        )
                    )
                    active.run = latest_run
                if latest_run.status in {"queued", "preparing", "running", "waiting"}:
                    await self._update_run(
                        active,
                        status=final_status,
                        keep_lease=False,
                        error=error_msg,
                        final_message=final_text[:MAX_WAKEUP_SUMMARY_CHARS] or None,
                        wakeup_delivered=(
                            True if wakeup_delivered else latest_run.wakeup_delivered
                        ),
                    )
                elif wakeup_delivered and not latest_run.wakeup_delivered:
                    await self._update_run(
                        active,
                        status=latest_run.status,
                        keep_lease=False,
                        error=latest_run.error,
                        final_message=latest_run.final_message,
                        wakeup_delivered=True,
                    )
