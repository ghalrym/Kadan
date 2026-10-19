import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
from kadan_tracing import (
    TEXT_ATTRIBUTES,
    PrivateExporter,
    RequestTracing,
    carrier,
    configure,
    span,
    traced_job,
)
from opentelemetry import trace
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode


@pytest.fixture
def recording():
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with patch(
        "opentelemetry.trace.get_tracer", return_value=provider.get_tracer("unit")
    ):
        yield exporter
    provider.shutdown()


def test_disabled_creates_no_exporter(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    with patch("kadan_tracing.OTLPSpanExporter") as exporter:
        assert configure("unit") is None
        exporter.assert_not_called()


def test_kill_switch_creates_no_exporter(monkeypatch):
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://unreachable:4318/v1/traces"
    )
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    with patch("kadan_tracing.OTLPSpanExporter") as exporter:
        assert configure("unit") is None
        exporter.assert_not_called()


def test_private_export_boundary_removes_contents_and_error_descriptions():
    destination = Mock()
    source = ReadableSpan(
        name="https://private/prompt",
        resource=Resource({"service.name": "unit"}),
        attributes={
            "url.full": "secret",
            "http.route": "/items/{item_id}",
            "prompt": "secret",
            "http.response.status_code": 500,
        },
        events=[Mock()],
        links=[Mock()],
        status=Status(StatusCode.ERROR, "private error"),
    )
    PrivateExporter(destination).export([source])
    exported = destination.export.call_args.args[0][0]
    assert exported.name == "HTTP request"
    assert dict(exported.attributes) == {
        "http.route": "/items/{item_id}",
        "http.response.status_code": 500,
    }
    assert not exported.events and not exported.links
    assert exported.status.status_code is StatusCode.ERROR
    assert exported.status.description is None


def test_span_error_contains_no_exception_contents(recording):
    with pytest.raises(ValueError):
        with span("kadan.unit"):
            raise ValueError("private prompt")
    result = recording.get_finished_spans()[0]
    assert result.status.status_code is StatusCode.ERROR
    assert result.status.description is None
    assert not result.events


def test_traceparent_propagates_without_baggage(recording):
    with span("kadan.parent") as parent:
        saved = carrier()
    assert set(saved) == {"traceparent"}
    with span("kadan.child", parent=saved) as child:
        assert child.get_span_context().trace_id == parent.get_span_context().trace_id
    assert (
        recording.get_finished_spans()[-1].parent.span_id
        == parent.get_span_context().span_id
    )


def test_async_job_restores_durable_context_and_measures_queue_wait(recording):
    with span("kadan.enqueue") as enqueue:
        saved = carrier()
    job = SimpleNamespace(
        trace_parent=saved["traceparent"],
        status="queued",
        created_at=datetime.now(UTC) - timedelta(seconds=2),
    )
    observed = []

    @traced_job("kadan.process")
    async def process(self, value):
        observed.append(trace.get_current_span().get_span_context().trace_id)
        return value

    assert asyncio.run(process(None, job)) is job
    assert observed == [enqueue.get_span_context().trace_id]
    queue = next(
        item
        for item in recording.get_finished_spans()
        if item.name == "kadan.queue.wait"
    )
    assert queue.end_time - queue.start_time >= 2_000_000_000


def test_job_cancellation_propagates_and_has_safe_status(recording):
    job = SimpleNamespace(
        trace_parent=None, status="running", created_at=datetime.now(UTC)
    )

    @traced_job("kadan.process")
    async def process(self, value):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(process(None, job))
    result = recording.get_finished_spans()[-1]
    assert result.status.status_code is StatusCode.ERROR
    assert not result.events
    assert result.attributes["kadan.recovery"] is True


def test_request_span_covers_stream_and_uses_route_template(recording):
    sent = []

    async def application(scope, receive, send):
        scope["route"] = SimpleNamespace(path="/items/{item_id}")
        await send({"type": "http.response.start", "status": 200})
        assert not recording.get_finished_spans()
        await send(
            {"type": "http.response.body", "body": b"private", "more_body": False}
        )

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/items/private",
        "headers": [(b"authorization", b"secret")],
    }
    asyncio.run(RequestTracing(application)(scope, Mock(), send))
    result = recording.get_finished_spans()[0]
    assert result.attributes["http.route"] == "/items/{item_id}"
    assert "private" not in str(result.attributes)
    assert len(sent) == 2


def test_enabled_configuration_is_bounded_without_connecting(monkeypatch):
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://collector:4318/v1/traces"
    )
    monkeypatch.setenv("KADAN_TRACE_SAMPLE_RATIO", "0.25")
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    with (
        patch("kadan_tracing.OTLPSpanExporter") as exporter,
        patch("kadan_tracing.BatchSpanProcessor") as batch,
        patch("kadan_tracing.trace.set_tracer_provider"),
        patch("kadan_tracing.set_global_textmap"),
        patch("kadan_tracing.HTTPXClientInstrumentor") as httpx,
        patch("kadan_tracing._processor"),
    ):
        provider = configure("unit")
        exporter.assert_called_once_with(
            endpoint="http://collector:4318/v1/traces", timeout=1
        )
        assert batch.call_args.kwargs == {
            "max_queue_size": 512,
            "max_export_batch_size": 64,
            "schedule_delay_millis": 1000,
            "export_timeout_millis": 1000,
        }
        assert provider.resource.attributes == {"service.name": "unit"}
        httpx.return_value.instrument.assert_called_once_with(tracer_provider=provider)
        provider.shutdown()


def test_actual_httpx_span_exports_safe_method_status_and_propagates_context():
    destination = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource({"service.name": "unit"}), shutdown_on_exit=False
    )
    provider.add_span_processor(SimpleSpanProcessor(PrivateExporter(destination)))
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(201)

    client = httpx.Client(transport=httpx.MockTransport(respond))
    HTTPXClientInstrumentor.instrument_client(client, tracer_provider=provider)
    try:
        client.post(
            "http://internal/private?secret=content",
            content="private prompt",
            headers={"Authorization": "credential"},
        )
    finally:
        client.close()
        provider.shutdown()
    exported = destination.get_finished_spans()[0]
    assert exported.attributes == {
        "http.request.method": "POST",
        "http.response.status_code": 201,
    }
    assert "traceparent" in requests[0].headers
    assert "private" not in str(exported.attributes)
    assert "credential" not in str(exported.attributes)


def test_text_metric_export_is_fixed_numeric_and_finite():
    destination = Mock()
    source = ReadableSpan(
        name="kadan.native.prefill",
        resource=Resource({"service.name": "unit"}),
        attributes={
            "kadan.text.tokens": 9,
            "kadan.text.large_row_seconds": 0.5,
            "kadan.text.rows": "private",
            "kadan.text.packed_calls": True,
            "kadan.text.unsupported_seconds": float("nan"),
            "kadan.text.batch_calls": -1,
            "kadan.text.batch_rejections": 2**63,
            "kadan.text.arbitrary": 42,
        },
    )
    PrivateExporter(destination).export([source])
    result = destination.export.call_args.args[0][0]
    assert dict(result.attributes) == {
        "kadan.text.tokens": 9,
        "kadan.text.large_row_seconds": 0.5,
    }
    assert len(TEXT_ATTRIBUTES) + 3 <= 32
