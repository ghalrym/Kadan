"""Opt-in, bounded tracing shared by the independently packaged Python services."""

import asyncio
import os
from contextlib import contextmanager
from functools import wraps
from inspect import iscoroutinefunction

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

propagator = TraceContextTextMapPropagator()
# No URLs, identifiers, headers, model inputs, exception text, or event contents.
ATTRIBUTES = frozenset(
    {
        "http.request.method",
        "http.response.status_code",
        "http.route",
        "kadan.stage",
        "kadan.worker",
        "kadan.recovery",
        "kadan.kv_cache.cross_request_reuse",
    }
)


class PrivateExporter(SpanExporter):
    """Enforce the export boundary even for third-party HTTP instrumentation."""

    def __init__(self, exporter):
        self.exporter = exporter

    def export(self, spans):
        clean = []
        for span in spans:
            attributes = {
                key: value
                for key, value in span.attributes.items()
                if key in ATTRIBUTES
            }
            name = span.name if span.name.startswith("kadan.") else "HTTP request"
            clean.append(
                ReadableSpan(
                    name=name,
                    context=span.context,
                    parent=span.parent,
                    resource=span.resource,
                    attributes=attributes,
                    events=(),
                    links=(),
                    kind=span.kind,
                    status=Status(span.status.status_code),
                    start_time=span.start_time,
                    end_time=span.end_time,
                    instrumentation_scope=span.instrumentation_scope,
                )
            )
        return self.exporter.export(clean)

    def shutdown(self):
        self.exporter.shutdown()

    def force_flush(self, timeout_millis=30000):
        return self.exporter.force_flush(timeout_millis)


def configure(service: str):
    """No endpoint means no SDK, instrumentation, exporter thread, or network IO."""
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "")
    if not endpoint or os.getenv("OTEL_SDK_DISABLED", "false").lower() == "true":
        return None
    try:
        ratio = min(1.0, max(0.0, float(os.getenv("KADAN_TRACE_SAMPLE_RATIO", "0.1"))))
    except ValueError:
        ratio = 0.1
    provider = TracerProvider(
        resource=Resource({"service.name": service}),
        sampler=ParentBased(TraceIdRatioBased(ratio)),
        span_limits=SpanLimits(
            max_attributes=16, max_events=0, max_links=0, max_attribute_length=128
        ),
        shutdown_on_exit=False,
    )
    global _processor
    _processor = BatchSpanProcessor(
        PrivateExporter(OTLPSpanExporter(endpoint=endpoint, timeout=1)),
        max_queue_size=512,
        max_export_batch_size=64,
        schedule_delay_millis=1000,
        export_timeout_millis=1000,
    )
    provider.add_span_processor(_processor)
    set_global_textmap(propagator)
    trace.set_tracer_provider(provider)
    HTTPXClientInstrumentor().instrument(tracer_provider=provider)
    return provider


async def shutdown(provider):
    if provider is not None:
        await asyncio.to_thread(provider.shutdown)


def carrier() -> dict[str, str]:
    value = {}
    propagator.inject(value)
    # Deliberately never propagate baggage or tracestate.
    return {"traceparent": value["traceparent"]} if "traceparent" in value else {}


@contextmanager
def span(name, *, parent=None, attributes=None, start_time=None):
    context = propagator.extract(parent) if parent is not None else None
    with trace.get_tracer("kadan").start_as_current_span(
        name,
        context=context,
        attributes=attributes,
        start_time=start_time,
        record_exception=False,
        set_status_on_exception=False,
    ) as current:
        try:
            yield current
        except BaseException:
            current.set_status(Status(StatusCode.ERROR))
            raise


def traced(name):
    """Explicit coarse function boundaries; never inspect arguments or results."""

    def decorate(function):
        if not iscoroutinefunction(function):

            @wraps(function)
            def call_sync(*args, **kwargs):
                with span(name):
                    return function(*args, **kwargs)

            return call_sync

        @wraps(function)
        async def call(*args, **kwargs):
            with span(name):
                return await function(*args, **kwargs)

        return call

    return decorate


class RequestTracing:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        method = scope.get("method", "")
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
            method = "OTHER"
        headers = {
            key.decode("latin1"): value.decode("latin1")
            for key, value in scope.get("headers", [])
            if key == b"traceparent"
        }
        with trace.get_tracer("kadan").start_as_current_span(
            "kadan.http.request",
            context=propagator.extract(headers),
            kind=SpanKind.SERVER,
            attributes={"http.request.method": method},
            record_exception=False,
            set_status_on_exception=False,
        ) as current:

            async def traced_send(message):
                if message["type"] == "http.response.start":
                    status = message["status"]
                    current.set_attribute("http.response.status_code", status)
                    context = current.get_span_context()
                    if context.is_valid:
                        message = {
                            **message,
                            "headers": [
                                *message.get("headers", []),
                                (
                                    b"x-kadan-trace-id",
                                    f"{context.trace_id:032x}".encode(),
                                ),
                            ],
                        }
                    if status >= 500:
                        current.set_status(Status(StatusCode.ERROR))
                await send(message)

            try:
                await self.app(scope, receive, traced_send)
            except BaseException:
                current.set_status(Status(StatusCode.ERROR))
                raise
            finally:
                route = scope.get("route")
                if route is not None and hasattr(route, "path"):
                    current.set_attribute("http.route", route.path)
                    current.update_name(f"kadan.http.{method} {route.path}")


def traced_job(name):
    """Restore the durable enqueue context, including after process restart."""

    def decorate(function):
        @wraps(function)
        async def call(self, job, *args, **kwargs):
            parent = {"traceparent": job.trace_parent} if job.trace_parent else {}
            queued = job.status == "queued"
            with span(
                name,
                parent=parent,
                attributes={"kadan.recovery": not queued},
                start_time=int(job.created_at.timestamp() * 1_000_000_000)
                if queued
                else None,
            ):
                if queued:
                    with span(
                        "kadan.queue.wait",
                        start_time=int(job.created_at.timestamp() * 1_000_000_000),
                    ):
                        pass
                return await function(self, job, *args, **kwargs)

        return call

    return decorate


# The API owns one exporter queue. Worker resources label the origin of the
# measured intervals; native code never waits for an OTLP collector.
_processor = None
_relays = {}


def worker_tracer(kind):
    if _processor is None:
        return trace.get_tracer("kadan")
    if kind not in _relays:
        provider = TracerProvider(
            resource=Resource({"service.name": "hikari-worker-" + kind}),
            shutdown_on_exit=False,
            span_limits=SpanLimits(max_attributes=16, max_events=0, max_links=0),
        )
        provider.add_span_processor(_processor)
        _relays[kind] = provider.get_tracer("kadan.native")
    return _relays[kind]
