"""Job spans and native phase spans relayed over the authenticated worker channel."""

from typing import TYPE_CHECKING, Literal

from kadan_tracing import propagator, worker_tracer
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, Field, ValidationError

if TYPE_CHECKING:
    from api.services.inference_worker import WorkerEvent
    from api.services.worker_hub import Worker


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
        if worker.kind == "llm" and event.phase == "prefill":
            # Engine::begin_request resets recurrent/KV state for every request.
            attributes["kadan.kv_cache.cross_request_reuse"] = False
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
