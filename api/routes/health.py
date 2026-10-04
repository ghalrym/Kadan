from typing import Literal
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


@router.get("/health", tags=["Health"])
def health() -> HealthResponse:
    return HealthResponse()
