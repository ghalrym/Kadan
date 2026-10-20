from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ...config import ServiceError

router = APIRouter()


class ComputerStatus(BaseModel):
    state: Literal["ready", "unavailable", "not_configured"]
    message: str


@router.get("/api/computer")
async def computer_status(request: Request) -> ComputerStatus:
    sandbox = request.app.state.services.sandbox
    if not sandbox.configured:
        return ComputerStatus(
            state="not_configured",
            message="No computer is configured. Set KADAN_SANDBOX_URL.",
        )
    try:
        await sandbox.context()
    except ServiceError as error:
        return ComputerStatus(state="unavailable", message=str(error))
    return ComputerStatus(state="ready", message="Ready")
