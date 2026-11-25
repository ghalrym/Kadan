"""Background watcher that wakes Kadan conversations when T3 Code threads finish or ask questions."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid5

from ..api.config import ServiceError
from ..api.pydantic_models.mcp import McpServerRecord, T3ThreadWatch

if TYPE_CHECKING:
    from ..api.services import LocalServices

logger = logging.getLogger(__name__)
POLL_SECONDS = 5
ACTIVE_STATUSES = frozenset({"preparing", "queued", "starting", "running", "waiting"})
TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "interrupted", "cancelled", "rolled_back", "idle"}
)
MAX_SUMMARY_CHARS = 12000


def _extract_pending_ids(result: dict[str, Any]) -> list[str]:
    candidates = (
        result.get("requestIds")
        or result.get("requests")
        or result.get("items")
        or result.get("pendingRequests")
        or []
    )
    ids: list[str] = []
    if isinstance(candidates, list):
        for entry in candidates:
            if isinstance(entry, str) and entry.strip():
                ids.append(entry.strip())
            elif isinstance(entry, dict):
                rid = entry.get("requestId") or entry.get("id")
                if isinstance(rid, str) and rid.strip():
                    ids.append(rid.strip())
    return ids


def _extract_run_status(
    thread_data: dict[str, Any], target_run_id: str | None
) -> str | None:
    runs = thread_data.get("runs")
    if isinstance(runs, list) and runs:
        if target_run_id:
            for run in runs:
                if isinstance(run, dict) and (
                    run.get("id") == target_run_id or run.get("runId") == target_run_id
                ):
                    st = run.get("status")
                    if isinstance(st, str):
                        return st.lower()
        latest = runs[0] if isinstance(runs[0], dict) else None
        if latest and isinstance(latest.get("status"), str):
            return latest["status"].lower()
    for key in ("status", "runStatus", "state"):
        val = thread_data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
    thread_obj = thread_data.get("thread")
    if isinstance(thread_obj, dict):
        for key in ("status", "runStatus", "state"):
            val = thread_obj.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip().lower()
    return None


def _extract_latest_reply(thread_data: dict[str, Any]) -> str:
    items = (
        thread_data.get("messages")
        or thread_data.get("items")
        or (
            thread_data["thread"].get("messages")
            if isinstance(thread_data.get("thread"), dict)
            else None
        )
        or []
    )
    if not isinstance(items, list):
        return ""
    for item in items:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or item.get("kind") or "").lower()
        if role in {"assistant", "agent", "assistant_message", "message"}:
            text = item.get("text") or item.get("content") or item.get("message")
            if isinstance(text, str) and text.strip():
                return text.strip()[:MAX_SUMMARY_CHARS]
    return ""


class T3ThreadEvents:
    """Polls durable T3 Code thread watches and queues idempotent Tsuchi event runs."""

    def __init__(self, services: LocalServices):
        self.services = services
        self.runner: asyncio.Task | None = None

    def start(self) -> None:
        self.runner = asyncio.create_task(self.run(), name="t3-thread-events")

    async def close(self) -> None:
        if self.runner:
            self.runner.cancel()
            await asyncio.gather(self.runner, return_exceptions=True)

    async def run(self) -> None:
        while True:
            try:
                await self.poll_once()
            except ServiceError as error:
                logger.warning("T3 thread watcher poll failed: %s", error)
            except Exception:
                logger.exception("Unexpected error in T3 thread watcher")
            await asyncio.sleep(POLL_SECONDS)

    async def poll_once(self) -> None:
        raw_watches = await self.services.json_request(
            "Tsuchi", "GET", "/api/t3-watches", params={"status": "watching"}
        )
        watches = [T3ThreadWatch.model_validate(item) for item in raw_watches]
        if not watches:
            return
        servers_by_id: dict[UUID, McpServerRecord] = {}
        for server in await self.services.mcp.list_servers():
            servers_by_id[server.id] = server

        for watch in watches:
            server = servers_by_id.get(watch.server_id)
            if server is None:
                await self._update_watch(watch, status="dropped")
                continue
            if not server.enabled or server.status != "connected":
                continue
            try:
                await self._check_watch(server, watch)
            except ServiceError as error:
                logger.warning(
                    "Could not check T3 thread %s: %s", watch.thread_id, error
                )

    async def _check_watch(
        self, server: McpServerRecord, watch: T3ThreadWatch
    ) -> None:
        tool_names = {tool.name for tool in server.tools}
        env_args = (
            {"environmentId": watch.environment_id} if watch.environment_id else {}
        )

        # 1. Check for pending user questions first so the agent can relay them in chat.
        if "t3_pending_request" in tool_names:
            try:
                pending_list = await self.services.mcp.call_tool(
                    server,
                    "t3_pending_request",
                    {"action": "list", "threadId": watch.thread_id, **env_args},
                )
            except ServiceError:
                pending_list = {}
            if isinstance(pending_list, dict) and "error" not in pending_list:
                pending_ids = _extract_pending_ids(pending_list)
                if pending_ids:
                    if pending_ids[0] != watch.last_pending_request_id:
                        req_id = pending_ids[0]
                        detail: dict[str, Any] = {"requestId": req_id}
                        try:
                            read_res = await self.services.mcp.call_tool(
                                server,
                                "t3_pending_request",
                                {
                                    "action": "read",
                                    "threadId": watch.thread_id,
                                    "requestId": req_id,
                                    **env_args,
                                },
                            )
                            if isinstance(read_res, dict) and "error" not in read_res:
                                detail = read_res
                        except ServiceError:
                            pass
                        delivered = await self._queue_event(
                            watch,
                            event_id=uuid5(watch.id, f"t3-q-{req_id}"),
                            payload={
                                "event_type": "pending_question",
                                "thread_id": watch.thread_id,
                                "thread_link": watch.thread_link,
                                "thread_title": watch.thread_title,
                                "request_id": req_id,
                                "question": detail,
                            },
                        )
                        await self._update_watch(
                            watch,
                            status="watching" if delivered else "dropped",
                            last_pending_request_id=req_id,
                        )
                    return

        # 2. Check thread run status.
        if "t3_thread_read" not in tool_names:
            return
        read_res = await self.services.mcp.call_tool(
            server,
            "t3_thread_read",
            {
                "threadId": watch.thread_id,
                "view": "messages",
                "limit": 5,
                **env_args,
            },
        )
        if not isinstance(read_res, dict):
            return
        if "error" in read_res:
            err_text = str(read_res["error"]).lower()
            if "not found" in err_text or "404" in err_text:
                await self._update_watch(watch, status="dropped")
            return

        status = _extract_run_status(read_res, watch.run_id)
        if status is None or status in ACTIVE_STATUSES:
            return
        if status not in TERMINAL_STATUSES:
            return

        link = (
            read_res.get("link")
            if isinstance(read_res.get("link"), str)
            else watch.thread_link
        )
        title = (
            read_res.get("title")
            if isinstance(read_res.get("title"), str)
            else watch.thread_title
        )
        summary = _extract_latest_reply(read_res)
        delivered = await self._queue_event(
            watch,
            event_id=uuid5(watch.id, "t3-done"),
            payload={
                "event_type": "thread_settled",
                "status": status,
                "thread_id": watch.thread_id,
                "thread_link": link,
                "thread_title": title,
                "message": summary,
            },
        )
        await self._update_watch(
            watch,
            status="delivered" if delivered else "dropped",
            thread_link=link,
            thread_title=title,
            last_pending_request_id=watch.last_pending_request_id,
        )

    async def _queue_event(
        self, watch: T3ThreadWatch, *, event_id: UUID, payload: dict[str, Any]
    ) -> bool:
        try:
            await self.services.json_request(
                "Tsuchi",
                "POST",
                f"/api/agents/{watch.origin_agent_key}/resources/"
                f"{watch.origin_conversation_id}/event-runs",
                json={
                    "id": str(event_id),
                    "origin_run_id": str(watch.origin_run_id),
                    "event": {
                        "kind": "t3",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                },
            )
            return True
        except ServiceError as error:
            if error.status == 404 or (
                error.status == 409
                and str(error) == "Restore this item from trash before editing it"
            ):
                logger.warning(
                    "T3 thread event for %s was dropped: %s", watch.thread_id, error
                )
                return False
            raise

    async def _update_watch(
        self,
        watch: T3ThreadWatch,
        *,
        status: str,
        thread_link: str | None = None,
        thread_title: str | None = None,
        last_pending_request_id: str | None = None,
    ) -> None:
        await self.services.json_request(
            "Tsuchi",
            "PUT",
            f"/api/t3-watches/{watch.id}",
            json={
                "status": status,
                "run_id": watch.run_id,
                "thread_link": thread_link
                if thread_link is not None
                else watch.thread_link,
                "thread_title": thread_title
                if thread_title is not None
                else watch.thread_title,
                "last_pending_request_id": last_pending_request_id
                if last_pending_request_id is not None
                else watch.last_pending_request_id,
                "expected_version": watch.version,
            },
        )
