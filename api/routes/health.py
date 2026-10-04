from typing import Literal
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


@router.get("/health", tags=["Health"], operation_id="health")
def health() -> HealthResponse:
    return HealthResponse()
