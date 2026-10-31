from uuid import UUID

from fastapi import APIRouter, Request

from ...coding_client import CodingClient
from ...pydantic_models.coding import CodexSetup, CodexStatus, CodexStopped, CodexThread

router = APIRouter(prefix="/api/computer/coding")


def client(request: Request) -> CodingClient:
    return CodingClient(request.app.state.services.sandbox)


@router.get("")
async def coding_status(request: Request) -> CodexStatus:
    return await client(request).status()


@router.get("/login/status")
async def codex_login_status(request: Request) -> CodexSetup:
    return await client(request).login_status()


@router.post("/login")
async def codex_login(request: Request) -> CodexSetup:
    return await client(request).login()


@router.post("/logout")
async def codex_logout(request: Request) -> CodexSetup:
    return await client(request).logout()


@router.get("/threads/{thread_id}")
async def codex_thread(request: Request, thread_id: UUID) -> CodexThread:
    return await client(request).thread(thread_id)


@router.post("/threads/{thread_id}/stop")
async def stop_codex_thread(request: Request, thread_id: UUID) -> CodexStopped:
    return await client(request).stop(thread_id)
