"""Job spans and native phase spans relayed over the authenticated worker channel."""

from typing import TYPE_CHECKING, Annotated, Literal

from kadan_tracing import propagator, worker_tracer
from kadan_tracing.cuda_metrics import CUDA_KERNELS
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from api.services.inference_worker import WorkerEvent
    from api.services.worker_hub import Worker


Duration = Annotated[float, Field(strict=True, ge=0, le=604800, allow_inf_nan=False)]
Count = Annotated[int, Field(strict=True, ge=0, lt=2**63)]


class TextPhaseMetrics(BaseModel):
    """Fixed numeric phase details; malformed details never invalidate a span."""

    model_config = ConfigDict(extra="forbid")

    tokens: Count
    projection_seconds: Duration
    attention_seconds: Duration
    prepare_seconds: Duration
    device_seconds: Duration
    host_wait_seconds: Duration
    recurrent_seconds: Duration
    projections: Count
    projection_batches: Count
    streamed_bytes: Count
    host_expert_bytes: Count
    packed_bytes: Count
    device_bytes: Count
    device_slices: Count
    ram_bytes: Count
    ram_slices: Count
    packed_calls: Count
    host_calls: Count
    large_row_calls: Count
    unsupported_calls: Count
    rows: Count
    batch_calls: Count
    batch_rejections: Count
    large_row_seconds: Duration
    unsupported_seconds: Duration


class CudaPhaseMetrics(BaseModel):
    """Bounded custom-kernel event times and host API costs, never payloads."""

    model_config = ConfigDict(extra="forbid")

    kernels: list[tuple[Count, Duration, Count, Duration]] = Field(
        min_length=len(CUDA_KERNELS), max_length=len(CUDA_KERNELS)
    )
    copies: list[tuple[Count, Duration]] = Field(min_length=5, max_length=5)
    sync_calls: Count
    sync_seconds: Duration
    dropped: Count
    unavailable: Count
    shared_batch_size: Annotated[int, Field(strict=True, ge=1, le=32)]

    def attributes(self) -> dict[str, int | float]:
        result = {
            "kadan.cuda.dropped": self.dropped,
            "kadan.cuda.unavailable": self.unavailable,
            "kadan.cuda.shared_batch_size": self.shared_batch_size,
        }
        for name, values in zip(CUDA_KERNELS, self.kernels, strict=True):
            if values[0] or values[2]:
                for field, value in zip(
                    ("calls", "host_seconds", "measured", "gpu_seconds"),
                    values,
                    strict=True,
                ):
                    result[f"kadan.cuda.{name}.{field}"] = value
        for kind, (calls, seconds) in enumerate(self.copies):
            if calls:
                result[f"kadan.cuda.copy{kind}.calls"] = calls
                result[f"kadan.cuda.copy{kind}.host_seconds"] = seconds
        if self.sync_calls:
            result["kadan.cuda.sync.calls"] = self.sync_calls
            result["kadan.cuda.sync.host_seconds"] = self.sync_seconds
        return result


class NativeSpan(BaseModel):
    phase: Literal[
        "prepare",
        "load",
        "admission",
        "execute",
        "prefill",
        "decode",
        "audio_frontend",
        "audio_encode",
    ]
    start: int = Field(gt=0)
    end: int = Field(gt=0)
    failed: bool = False
    traceparent: str
    metrics: object = None
    cuda: object = None


class JobTracing:
    def __init__(self) -> None:
        self.span = trace.get_tracer("kadan").start_span("kadan.inference")
        self.context = trace.set_span_in_context(self.span)
        self.parent = {}
        propagator.inject(self.parent, context=self.context)
        self.parent = (
            {"traceparent": self.parent["traceparent"]}
            if "traceparent" in self.parent
            else {}
        )
        self.queued = {}
        self.closed = False

    def event(self, worker: "Worker", event: "WorkerEvent") -> None:
        if self.closed or event.trace_unix_ns is None:
            return
        if event.state == "queued":
            self.queued.setdefault(worker.id, event.trace_unix_ns)
        elif event.state in {"preparing", "cancelled", "failed", "rejected"}:
            start = self.queued.pop(worker.id, None)
            if start is not None and event.trace_unix_ns >= start:
                stage = worker_tracer(worker.kind).start_span(
                    "kadan.queue.wait", context=self.context, start_time=start
                )
                stage.end(end_time=event.trace_unix_ns)

    def native(self, worker: "Worker", message: object) -> None:
        if self.closed or not self.span.is_recording():
            return
        try:
            event = NativeSpan.model_validate(message)
        except ValidationError:
            return
        if (
            event.traceparent != self.parent.get("traceparent")
            or event.end < event.start
        ):
            return
        # Bound corrupted timestamps/durations without affecting job processing.
        if event.end - event.start > 7 * 24 * 3600 * 1_000_000_000:
            return
        attributes = {"kadan.stage": event.phase, "kadan.worker": worker.kind}
        if worker.kind == "llm" and event.phase in {"prefill", "decode"}:
            try:
                metrics = TextPhaseMetrics.model_validate(event.metrics)
            except ValidationError:
                pass
            else:
                attributes.update(
                    {
                        "kadan.text." + key: value
                        for key, value in metrics.model_dump().items()
                    }
                )
        try:
            cuda = CudaPhaseMetrics.model_validate(event.cuda)
        except ValidationError:
            pass
        else:
            attributes.update(cuda.attributes())
        current = worker_tracer(worker.kind).start_span(
            "kadan.native." + event.phase,
            context=self.context,
            start_time=event.start,
            attributes=attributes,
        )
        if event.failed:
            current.set_status(Status(StatusCode.ERROR))
        current.end(end_time=event.end)

    def end(self, failed: bool = False) -> None:
        if self.closed:
            return
        self.closed = True
        if failed:
            self.span.set_status(Status(StatusCode.ERROR))
        self.span.end()
