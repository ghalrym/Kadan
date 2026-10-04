from typing import Literal
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/v1/metrics", tags=["Monitoring"])


class ResourceMeter(BaseModel):
    label: str
    used: float
    total: float


class MetricsResponse(BaseModel):
    status: Literal["Online"]
    resources: list[ResourceMeter]
    requests_per_minute: int
    p50_latency_seconds: float
    error_rate_percent: float
    queued_jobs: int


@router.get("", operation_id="getMetrics")
def get_metrics() -> MetricsResponse:
    return MetricsResponse(status="Online", resources=[
        ResourceMeter(label="GPU 0 · VRAM", used=18.6, total=24),
        ResourceMeter(label="GPU 1 · VRAM", used=14.8, total=24),
        ResourceMeter(label="RAM", used=196.0, total=512),
    ], requests_per_minute=14, p50_latency_seconds=1.68, error_rate_percent=14.3, queued_jobs=2)
