"""High-level Work service client bridging Kadan API/tools, Tsuchi storage, folder browser, and harnesses."""

from __future__ import annotations

from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid5

from ..api.config import ServiceError
from ..api.pydantic_models.work import (
    FilesystemBrowseResponse,
    WorkActivityItem,
    WorkHarnessAuthPublic,
    WorkHarnessModelInfo,
    WorkHarnessPublic,
    WorkHarnessSignInSubmit,
    WorkPendingRespondRequest,
    WorkProject,
    WorkProjectCreate,
    WorkProjectPatch,
    WorkRun,
    WorkRunCreate,
    WorkThread,
    WorkThreadCreate,
    WorkThreadDetail,
    WorkThreadOrganizeRequest,
    WorkThreadPatch,
)
from .contract import HarnessAuthStatus, HarnessError
from .filesystem import browse_directory, inspect_directory
from .registry import HarnessRegistry, default_registry

if TYPE_CHECKING:
    from ..api.services import LocalServices


def auto_thread_title(
    prompt: str | None, *, has_images: bool = False, max_chars: int = 60
) -> str:
    """Derive a concise readable thread title from the first prompt."""
    if prompt:
        for raw_line in prompt.splitlines():
            line = " ".join(raw_line.strip().lstrip("#*-•> ").split())
            if not line:
                continue
            if len(line) <= max_chars:
                return line
            cut = line[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:-")
            return f"{cut or line[:max_chars]}…"
    return "Image task" if has_images else "New thread"


def work_thread_url(thread_id: UUID) -> str:
    """Canonical in-app anchor link for opening a Work thread inside Kadan."""
    return f"#work/{thread_id}"


def work_thread_markdown_link(thread_id: UUID, title: str) -> str:
    safe_title = title.replace("[", "(").replace("]", ")")
    return f"[{safe_title}]({work_thread_url(thread_id)})"


def _public_auth(auth: HarnessAuthStatus) -> WorkHarnessAuthPublic:
    return WorkHarnessAuthPublic(
        status=auth.status,
        login_method=auth.login_method,
        authorize_url=auth.authorize_url,
        user_code=auth.user_code,
        needs_code=auth.needs_code,
        account_label=auth.account_label,
        message=auth.message,
    )


class WorkClient:
    """Manages native Work projects, threads, runs, and coding harnesses."""

    def __init__(
        self,
        services: LocalServices,
        *,
        registry: HarnessRegistry | None = None,
    ):
        self.services = services
        self.registry = registry or default_registry()

    # ── Filesystem picker ────────────────────────────────────────────────────

    async def browse_filesystem(
        self, path: str | None = None
    ) -> FilesystemBrowseResponse:
        raw = await browse_directory(path)
        return FilesystemBrowseResponse.model_validate(raw)

    # ── Projects ─────────────────────────────────────────────────────────────

    async def list_projects(self) -> list[WorkProject]:
        rows = await self.services.json_request("Tsuchi", "GET", "/api/work/projects")
        return [WorkProject.model_validate(row) for row in rows]

    async def get_project(self, project_id: UUID) -> WorkProject:
        raw = await self.services.json_request(
            "Tsuchi", "GET", f"/api/work/projects/{project_id}"
        )
        return WorkProject.model_validate(raw)

    async def create_project(self, data: WorkProjectCreate) -> WorkProject:
        info = await inspect_directory(data.root_path)
        root_path = info["path"]
        title = (data.title or "").strip() or info["name"]
        raw = await self.services.json_request(
            "Tsuchi",
            "POST",
            "/api/work/projects",
            json={
                "title": title,
                "root_path": root_path,
            },
        )
        return WorkProject.model_validate(raw)

    async def update_project(
        self, project_id: UUID, patch: WorkProjectPatch
    ) -> WorkProject:
        current = await self.get_project(project_id)
        payload = {
            "title": patch.title if patch.title is not None else current.title,
            "root_path": current.root_path,
            "expected_version": patch.expected_version,
        }
        raw = await self.services.json_request(
            "Tsuchi", "PUT", f"/api/work/projects/{project_id}", json=payload
        )
        return WorkProject.model_validate(raw)

    async def delete_project(self, project_id: UUID, *, force: bool = False) -> None:
        await self.services.json_request(
            "Tsuchi",
            "DELETE",
            f"/api/work/projects/{project_id}",
            params={"force": "true"} if force else None,
        )

    # ── Harnesses ────────────────────────────────────────────────────────────

    async def list_harnesses(self) -> list[WorkHarnessPublic]:
        results: list[WorkHarnessPublic] = []
        for adapter in self.registry.list():
            avail = await adapter.check_availability(self.services)
            results.append(
                WorkHarnessPublic(
                    id=avail.harness_id,
                    display_name=avail.display_name,
                    description=avail.description,
                    installed=avail.installed,
                    auth=_public_auth(avail.auth),
                    models=[
                        WorkHarnessModelInfo(
                            id=m.id,
                            label=m.label,
                            description=m.description,
                            options=m.options,
                        )
                        for m in avail.models
                    ],
                    default_model=avail.default_model,
                    supports_steer=avail.supports_steer,
                    supports_resume=avail.supports_resume,
                    supports_approvals=avail.supports_approvals,
                    supports_questions=avail.supports_questions,
                    supports_images=avail.supports_images,
                )
            )
        return results

    async def start_harness_sign_in(
        self, harness_id: str, payload: dict[str, Any] | None = None
    ) -> WorkHarnessAuthPublic:
        adapter = self.registry.get(harness_id)
        try:
            auth = await adapter.start_sign_in(self.services, payload)
        except HarnessError as exc:
            raise ServiceError(str(exc), 422, "harness_sign_in_failed") from exc
        return _public_auth(auth)

    async def get_harness_sign_in(self, harness_id: str) -> WorkHarnessAuthPublic:
        adapter = self.registry.get(harness_id)
        return _public_auth(await adapter.sign_in_status(self.services))

    async def submit_harness_sign_in(
        self, harness_id: str, data: WorkHarnessSignInSubmit
    ) -> WorkHarnessAuthPublic:
        adapter = self.registry.get(harness_id)
        try:
            auth = await adapter.submit_sign_in(
                self.services, data.model_dump(exclude_none=True)
            )
        except HarnessError as exc:
            raise ServiceError(str(exc), 422, "harness_sign_in_failed") from exc
        return _public_auth(auth)

    async def sign_out_harness(self, harness_id: str) -> WorkHarnessAuthPublic:
        adapter = self.registry.get(harness_id)
        return _public_auth(await adapter.sign_out(self.services))

    # ── Threads, runs, activity & pending requests ───────────────────────────

    async def list_threads(
        self,
        *,
        bucket: str | None = None,
        project_id: UUID | None = None,
        q: str = "",
        limit: int = 200,
    ) -> list[WorkThread]:
        params: dict[str, Any] = {"limit": limit}
        if bucket:
            params["bucket"] = bucket
        if project_id is not None:
            params["project_id"] = str(project_id)
        if q:
            params["q"] = q
        rows = await self.services.json_request(
            "Tsuchi", "GET", "/api/work/threads", params=params
        )
        return [WorkThread.model_validate(row) for row in rows]

    async def get_thread(self, thread_id: UUID) -> WorkThread:
        raw = await self.services.json_request(
            "Tsuchi", "GET", f"/api/work/threads/{thread_id}"
        )
        return WorkThread.model_validate(raw)

    async def get_thread_detail(
        self, thread_id: UUID, *, activity_limit: int = 300
    ) -> WorkThreadDetail:
        thread = await self.get_thread(thread_id)
        project: WorkProject | None = None
        with suppress(ServiceError):
            project = await self.get_project(thread.project_id)
        runs_raw = await self.services.json_request(
            "Tsuchi", "GET", f"/api/work/threads/{thread_id}/runs"
        )
        activity_raw = await self.services.json_request(
            "Tsuchi",
            "GET",
            f"/api/work/threads/{thread_id}/activity",
            params={"limit": activity_limit},
        )
        return WorkThreadDetail(
            thread=thread,
            project=project,
            runs=[WorkRun.model_validate(r) for r in runs_raw],
            activity=[WorkActivityItem.model_validate(a) for a in activity_raw],
        )

    async def create_thread(
        self,
        data: WorkThreadCreate,
        *,
        origin_agent_key: str | None = None,
        origin_conversation_id: UUID | None = None,
        origin_run_id: UUID | None = None,
    ) -> tuple[WorkThread, WorkRun | None]:
        adapter = self.registry.get(data.harness)
        project = await self.get_project(data.project_id)
        resolved_model = data.model
        if not resolved_model:
            avail = await adapter.check_availability(self.services)
            resolved_model = avail.default_model

        resolved_title = (data.title or "").strip() or auto_thread_title(
            data.prompt, has_images=bool(data.images)
        )
        thread_payload: dict[str, Any] = {
            "client_request_id": data.client_request_id,
            "project_id": str(project.id),
            "title": resolved_title,
            "harness": data.harness,
            "model": resolved_model,
            "model_options": data.model_options,
            "origin_agent_key": origin_agent_key,
            "origin_conversation_id": (
                str(origin_conversation_id) if origin_conversation_id else None
            ),
            "origin_run_id": str(origin_run_id) if origin_run_id else None,
        }
        created = WorkThread.model_validate(
            await self.services.json_request(
                "Tsuchi", "POST", "/api/work/threads", json=thread_payload
            )
        )

        run: WorkRun | None = None
        if (data.prompt and data.prompt.strip()) or data.images:
            run_client_id = (
                f"{data.client_request_id}:initial-run"
                if data.client_request_id
                else str(uuid5(created.id, "initial-run"))
            )
            prompt_text = (
                data.prompt or ""
            ).strip() or "Inspect the attached image(s)."
            run = await self.send_message(
                created.id,
                WorkRunCreate(
                    prompt=prompt_text,
                    mode="send",
                    images=data.images,
                    client_request_id=run_client_id,
                ),
            )
            created = await self.get_thread(created.id)
        return created, run

    async def update_thread(
        self, thread_id: UUID, patch: WorkThreadPatch
    ) -> WorkThread:
        if patch.harness is not None:
            self.registry.get(patch.harness)
        payload = patch.model_dump(mode="json", exclude_unset=True)
        raw = await self.services.json_request(
            "Tsuchi", "PUT", f"/api/work/threads/{thread_id}", json=payload
        )
        return WorkThread.model_validate(raw)

    async def delete_thread(self, thread_id: UUID) -> None:
        with suppress(ServiceError):
            thread = await self.get_thread(thread_id)
            adapter = self.registry.get(thread.harness)
            drop = getattr(adapter, "drop_session_state", None)
            if callable(drop):
                drop(thread_id)
        await self.services.json_request(
            "Tsuchi", "DELETE", f"/api/work/threads/{thread_id}"
        )

    async def organize_thread(
        self, thread_id: UUID, req: WorkThreadOrganizeRequest
    ) -> WorkThread:
        if req.action in {"default", "unsettle"}:
            bucket = "default"
            snoozed_until = None
        elif req.action == "settle":
            bucket = "settled"
            snoozed_until = None
        else:
            bucket = "snoozed"
            if req.snoozed_until is not None:
                snoozed_until = req.snoozed_until
            elif req.snooze_minutes is not None:
                snoozed_until = datetime.now(UTC) + timedelta(
                    minutes=req.snooze_minutes
                )
            else:
                snoozed_until = datetime.now(UTC) + timedelta(hours=1)
        payload: dict[str, Any] = {
            "bucket": bucket,
            "snoozed_until": snoozed_until.isoformat() if snoozed_until else None,
        }
        if req.expected_version is not None:
            payload["expected_version"] = req.expected_version
        raw = await self.services.json_request(
            "Tsuchi", "POST", f"/api/work/threads/{thread_id}/organize", json=payload
        )
        return WorkThread.model_validate(raw)

    async def send_message(self, thread_id: UUID, req: WorkRunCreate) -> WorkRun:
        raw = await self.services.json_request(
            "Tsuchi",
            "POST",
            f"/api/work/threads/{thread_id}/runs",
            json=req.model_dump(mode="json", exclude_none=True),
        )
        return WorkRun.model_validate(raw)

    async def interrupt_thread(self, thread_id: UUID) -> WorkThread:
        runs_raw = await self.services.json_request(
            "Tsuchi", "GET", f"/api/work/threads/{thread_id}/runs"
        )
        runs = [WorkRun.model_validate(r) for r in runs_raw]
        for run in runs:
            if run.status in {"queued", "preparing", "running", "waiting"}:
                with suppress(ServiceError):
                    await self.services.json_request(
                        "Tsuchi", "POST", f"/api/work/runs/{run.id}/cancel"
                    )
        return await self.get_thread(thread_id)

    async def respond_pending_request(
        self,
        thread_id: UUID,
        request_id: str,
        req: WorkPendingRespondRequest,
    ) -> WorkThread:
        raw = await self.services.json_request(
            "Tsuchi",
            "POST",
            f"/api/work/threads/{thread_id}/pending-requests/{request_id}/respond",
            json=req.model_dump(mode="json", exclude_none=True),
        )
        return WorkThread.model_validate(raw)

    async def list_activity(
        self,
        thread_id: UUID,
        *,
        after_position: int = -1,
        limit: int = 200,
    ) -> list[WorkActivityItem]:
        rows = await self.services.json_request(
            "Tsuchi",
            "GET",
            f"/api/work/threads/{thread_id}/activity",
            params={"after_position": after_position, "limit": limit},
        )
        return [WorkActivityItem.model_validate(row) for row in rows]

    async def wake_snoozed(self) -> list[WorkThread]:
        rows = await self.services.json_request(
            "Tsuchi", "POST", "/api/work/threads/wake-snoozed"
        )
        return [WorkThread.model_validate(row) for row in rows]
