"""HTTP API routes for Kadan native Work projects, threads, runs, activity, and harnesses."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Request

from ..pydantic_models.work import (
    FilesystemBrowseResponse,
    WorkActivityItem,
    WorkBucket,
    WorkHarnessAuthPublic,
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

router = APIRouter(prefix="/work", tags=["work"])


# ── Filesystem picker ────────────────────────────────────────────────────────


@router.get("/filesystem/browse", response_model=FilesystemBrowseResponse)
async def browse_filesystem(request: Request, path: str | None = None):
    return await request.app.state.services.work.browse_filesystem(path)


# ── Projects ─────────────────────────────────────────────────────────────────


@router.get("/projects", response_model=list[WorkProject])
async def list_projects(request: Request):
    return await request.app.state.services.work.list_projects()


@router.post("/projects", response_model=WorkProject, status_code=201)
async def create_project(request: Request, body: WorkProjectCreate):
    return await request.app.state.services.work.create_project(body)


@router.put("/projects/{project_id}", response_model=WorkProject)
async def update_project(request: Request, project_id: UUID, body: WorkProjectPatch):
    return await request.app.state.services.work.update_project(project_id, body)


@router.delete("/projects/{project_id}", status_code=204)
async def delete_project(request: Request, project_id: UUID, force: bool = False):
    await request.app.state.services.work.delete_project(project_id, force=force)


# ── Harnesses ────────────────────────────────────────────────────────────────


@router.get("/harnesses", response_model=list[WorkHarnessPublic])
async def list_harnesses(request: Request):
    return await request.app.state.services.work.list_harnesses()


@router.post("/harnesses/{harness}/sign-in", response_model=WorkHarnessAuthPublic)
async def start_harness_sign_in(request: Request, harness: str):
    return await request.app.state.services.work.start_harness_sign_in(harness)


@router.get("/harnesses/{harness}/sign-in", response_model=WorkHarnessAuthPublic)
async def get_harness_sign_in(request: Request, harness: str):
    return await request.app.state.services.work.get_harness_sign_in(harness)


@router.post(
    "/harnesses/{harness}/sign-in/submit", response_model=WorkHarnessAuthPublic
)
async def submit_harness_sign_in(
    request: Request, harness: str, body: WorkHarnessSignInSubmit
):
    return await request.app.state.services.work.submit_harness_sign_in(harness, body)


@router.delete("/harnesses/{harness}/sign-in", response_model=WorkHarnessAuthPublic)
async def sign_out_harness(request: Request, harness: str):
    return await request.app.state.services.work.sign_out_harness(harness)


# ── Threads ──────────────────────────────────────────────────────────────────


@router.get("/threads", response_model=list[WorkThread])
async def list_threads(
    request: Request,
    bucket: WorkBucket | None = None,
    project_id: UUID | None = None,
    q: str = "",
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
):
    return await request.app.state.services.work.list_threads(
        bucket=bucket, project_id=project_id, q=q, limit=limit
    )


@router.post("/threads", response_model=WorkThread, status_code=201)
async def create_thread(request: Request, body: WorkThreadCreate):
    thread, _ = await request.app.state.services.work.create_thread(body)
    return thread


@router.get("/threads/{thread_id}", response_model=WorkThreadDetail)
async def get_thread_detail(request: Request, thread_id: UUID):
    return await request.app.state.services.work.get_thread_detail(thread_id)


@router.put("/threads/{thread_id}", response_model=WorkThread)
async def update_thread(request: Request, thread_id: UUID, body: WorkThreadPatch):
    return await request.app.state.services.work.update_thread(thread_id, body)


@router.delete("/threads/{thread_id}", status_code=204)
async def delete_thread(request: Request, thread_id: UUID):
    await request.app.state.services.work.delete_thread(thread_id)


@router.post("/threads/{thread_id}/organize", response_model=WorkThread)
async def organize_thread(
    request: Request, thread_id: UUID, body: WorkThreadOrganizeRequest
):
    return await request.app.state.services.work.organize_thread(thread_id, body)


@router.post("/threads/{thread_id}/runs", response_model=WorkRun, status_code=201)
async def send_thread_message(request: Request, thread_id: UUID, body: WorkRunCreate):
    return await request.app.state.services.work.send_message(thread_id, body)


@router.post("/threads/{thread_id}/interrupt", response_model=WorkThread)
async def interrupt_thread(request: Request, thread_id: UUID):
    return await request.app.state.services.work.interrupt_thread(thread_id)


@router.post(
    "/threads/{thread_id}/pending-requests/{request_id}/respond",
    response_model=WorkThread,
)
async def respond_pending_request(
    request: Request,
    thread_id: UUID,
    request_id: str,
    body: WorkPendingRespondRequest,
):
    return await request.app.state.services.work.respond_pending_request(
        thread_id, request_id, body
    )


@router.get("/threads/{thread_id}/activity", response_model=list[WorkActivityItem])
async def list_thread_activity(
    request: Request,
    thread_id: UUID,
    after_position: int = -1,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
):
    return await request.app.state.services.work.list_activity(
        thread_id, after_position=after_position, limit=limit
    )
