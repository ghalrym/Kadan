from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    version: str = "0.1.0"
    app: str = "kadan-server"


@router.get("/api/health")
async def health(request: Request) -> Health:
    services = getattr(request.app.state, "services", None)
    version = (
        getattr(services.settings, "app_version", "0.1.0")
        if services is not None
        else "0.1.0"
    )
    return Health(version=version)
