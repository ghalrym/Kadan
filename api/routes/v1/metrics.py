import sys
from typing import Literal
from fastapi import APIRouter
from pydantic import BaseModel
from api.services.telemetry import telemetry

router = APIRouter(prefix='/v1/metrics', tags=['Monitoring'])


class ResourceMeter(BaseModel):
    label: str
    used: float
    total: float


class MetricsResponse(BaseModel):
    status: Literal['Online'] = 'Online'
    resources: list[ResourceMeter]
    resource_errors: list[str]
    memory_unit: Literal['GiB'] = 'GiB'
    started_at: str
    retention_limit: int
    retained_requests: int
    completed_requests: int
    active_requests: int
    requests_per_minute: int
    window_seconds: int = 60
    window_truncated: bool
    p50_latency_seconds: float | None
    error_rate_percent: float | None


def memory_meters():
    resources, errors = [], []
    try:
        with open('/proc/meminfo') as stream:
            values = {parts[0].rstrip(':'): int(parts[1]) * 1024
                      for line in stream if (parts := line.split()) and parts[0] in ('MemTotal:', 'MemAvailable:')}
        total, available = values['MemTotal'], values['MemAvailable']
        resources.append(ResourceMeter(label='Host RAM', used=max(0, total - available) / 1024**3, total=total / 1024**3))
    except (OSError, KeyError, ValueError):
        errors.append('Host RAM measurement unavailable')
    # Do not import optional heavyweight inference dependencies for dashboard polls.
    torch = sys.modules.get('torch')
    if torch is None:
        errors.append('GPU measurement unavailable until inference dependencies are loaded')
    else:
        try:
            if not torch.cuda.is_available():
                errors.append('CUDA GPU measurement unavailable')
            else:
                for device in range(torch.cuda.device_count()):
                    free, total = torch.cuda.mem_get_info(device)
                    resources.append(ResourceMeter(label=f'GPU {device} · VRAM',
                                                   used=max(0, total - free) / 1024**3, total=total / 1024**3))
        except Exception:
            errors.append('GPU measurement failed')
    return resources, errors


@router.get('', operation_id='getMetrics')
def get_metrics() -> MetricsResponse:
    resources, errors = memory_meters()
    return MetricsResponse(resources=resources, resource_errors=errors, **telemetry.metrics())
