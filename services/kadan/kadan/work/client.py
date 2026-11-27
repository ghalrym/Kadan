"""High-level Work service client bridging Kadan API/tools, Tsuchi storage, worktrees, and harnesses."""

from __future__ import annotations

from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
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
from .registry import HarnessRegistry, default_registry
from .worktree import WorktreeManager

if TYPE_CHECKING:
    from ..api.services import LocalServices


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
    """Manages native Work projects, threads, runs, harnesses, and worktrees."""

    def __init__(
        self,
        services: LocalServices,
        *,
        registry: HarnessRegistry | None = None,
        worktrees: WorktreeManager | None = None,
    ):
        self.services = services
        self.registry = registry or default_registry()
        self.worktrees = worktrees or WorktreeManager()

    # ── Filesystem picker ────────────────────────────────────────────────────

    async def browse_filesystem(
        self, path: str | None = None
    ) -> FilesystemBrowseResponse:
        raw = await self.worktrees.browse_directory(path)
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
        has_root = bool(data.root_path and data.root_path.strip())
        has_clone = bool(data.clone_url and data.clone_url.strip())
        if has_root == has_clone:
            raise ServiceError(
                "Choose either a local project folder or a repository URL to clone.",
                422,
                "invalid_project_source",
            )
        if has_clone:
            assert data.clone_url is not None
            (
                dest,
                detected_branch,
                detected_title,
            ) = await self.worktrees.clone_repository(data.clone_url, title=data.title)
            root_path = str(dest)
            title = (data.title or "").strip() or detected_title
            default_branch = (data.default_branch or "").strip() or detected_branch
        else:
            assert data.root_path is not None
            info = await self.worktrees.inspect_directory(data.root_path)
            root_path = str(info["path"])
            title = (data.title or "").strip() or str(info["name"])
            default_branch = (data.default_branch or "").strip() or str(
                info["default_branch"] or "main"
            )

        raw = await self.services.json_request(
            "Tsuchi",
            "POST",
            "/api/work/projects",
            json={
                "title": title,
                "root_path": root_path,
                "default_branch": default_branch,
                "setup_script": data.setup_script,
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
            "default_branch": (
                patch.default_branch
                if patch.default_branch is not None
                else current.default_branch
            ),
            "setup_script": (
                patch.setup_script
                if "setup_script" in patch.model_fields_set
                else current.setup_script
            ),
            "expected_version": patch.expected_version,
        }
        raw = await self.services.json_request(
            "Tsuchi", "PUT", f"/api/work/projects/{project_id}", json=payload
        )
        return WorkProject.model_validate(raw)

    async def delete_project(self, project_id: UUID, *, force: bool = False) -> None:
        if force:
            threads = await self.list_threads(project_id=project_id, limit=500)
            project = await self.get_project(project_id)
            for thread in threads:
                if thread.workspace_mode == "worktree" and thread.worktree_path:
                    with suppress(Exception):
                        await self.worktrees.remove_worktree(
                            Path(project.root_path),
                            Path(thread.worktree_path),
                            delete_branch=thread.branch,
                        )
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
        # Validate harness exists in registry and project exists in Tsuchi.
        adapter = self.registry.get(data.harness)
        project = await self.get_project(data.project_id)
        resolved_model = data.model
        if not resolved_model:
            avail = await adapter.check_availability(self.services)
            resolved_model = avail.default_model

        thread_payload: dict[str, Any] = {
            "client_request_id": data.client_request_id,
            "project_id": str(project.id),
            "title": data.title,
            "harness": data.harness,
            "model": resolved_model,
            "model_options": data.model_options,
            "workspace_mode": data.workspace_mode,
            "branch": data.branch,
            "base_ref": data.base_ref or project.default_branch,
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

        # Prepare worktree eagerly if requested and not yet created.
        if created.workspace_mode == "worktree" and not created.worktree_path:
            wt = await self.worktrees.create_worktree(
                Path(project.root_path),
                created.id,
                base_ref=created.base_ref or project.default_branch,
                branch=created.branch,
                setup_script=project.setup_script,
            )
            created = WorkThread.model_validate(
                await self.services.json_request(
                    "Tsuchi",
                    "PUT",
                    f"/api/work/threads/{created.id}",
                    json={
                        "worktree_path": str(wt.path),
                        "branch": wt.branch,
                        "base_ref": wt.base_ref,
                        "expected_version": created.version,
                    },
                )
            )

        run: WorkRun | None = None
        if data.prompt and data.prompt.strip():
            run_client_id = (
                f"{data.client_request_id}:initial-run"
                if data.client_request_id
                else str(uuid5(created.id, "initial-run"))
            )
            run = await self.send_message(
                created.id,
                WorkRunCreate(
                    prompt=data.prompt.strip(),
                    mode="send",
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
        thread = await self.get_thread(thread_id)
        if thread.workspace_mode == "worktree" and thread.worktree_path:
            with suppress(ServiceError, Exception):
                project = await self.get_project(thread.project_id)
                await self.worktrees.remove_worktree(
                    Path(project.root_path),
                    Path(thread.worktree_path),
                    delete_branch=thread.branch,
                )
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
