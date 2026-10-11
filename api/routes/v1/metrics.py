from typing import Literal
from fastapi import APIRouter
from pydantic import BaseModel
from api.services.telemetry import telemetry
from api.services.inference import inference

router = APIRouter(prefix='/v1/metrics', tags=['Monitoring'])


class ResourceMeter(BaseModel):
    """Observed host or device-wide memory in GiB, including other processes."""
    label: str
    used: float
    total: float


class InferenceProgress(BaseModel):
    job_id: str
    workload: str
    stage: str
    value: int
    observed_unix_ns: int


class MetricsResponse(BaseModel):
    """Process-local request statistics plus best-effort host/device memory samples.

    Null latency/error statistics mean no retained samples in the window. A
    truncated window covers retained records only; Online does not imply a
    loaded model or successful GPU inference."""
    inference_progress: InferenceProgress | None = None
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
    """Report host observations and the native worker's reservation ledger."""
    resources, errors = [], []
    try:
        with open('/proc/meminfo') as stream:
            values = {parts[0].rstrip(':'): int(parts[1]) * 1024
                      for line in stream if (parts := line.split()) and parts[0] in ('MemTotal:', 'MemAvailable:')}
        total, available = values['MemTotal'], values['MemAvailable']
        resources.append(ResourceMeter(label='Host RAM', used=max(0, total - available) / 1024**3, total=total / 1024**3))
    except (OSError, KeyError, ValueError):
        errors.append('Host RAM measurement unavailable')
    memory = inference.worker.memory
    if memory is None:
        errors.append('Native memory observations are not available yet')
    else:
        for device, (capacity, used) in enumerate(zip(memory.capacity[1:], memory.used[1:])):
            resources.append(ResourceMeter(label=f'GPU {device} · native reservations',
                used=used / 1024**3, total=capacity / 1024**3))
    return resources, errors


@router.get('', operation_id='getMetrics')
def get_metrics() -> MetricsResponse:
    """Combine a fresh memory sample with the bounded telemetry snapshot.

    This monitoring read is excluded from generation telemetry, so polling
    does not increase request counts or feed back into latency statistics."""
    resources, errors = memory_meters()
    return MetricsResponse(inference_progress=inference.progress, resources=resources, resource_errors=errors, **telemetry.metrics())
