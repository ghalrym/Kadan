from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    version: str = "0.1.0"
    app: str = "kadan-server"


@router.get("/api/health")
async def health() -> Health:
    return Health()
