import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi.telemetry import _runtime
from kadan_tracing import PrivateExporter
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

from api.server import app, lifespan


class LifespanTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_starts_fifo_and_shutdown_closes_owners(self):
        manager = Mock(start=AsyncMock(), close=AsyncMock())
        with patch("api.server.inference", manager):
            async with lifespan(app):
                manager.start.assert_awaited_once()
                manager.close.assert_not_awaited()
            manager.close.assert_awaited_once()

    async def test_lifespan_owns_filtered_hikari_provider_after_fastapi_startup(self):
        destination = InMemorySpanExporter()
        manager = Mock(start=AsyncMock(), close=AsyncMock())
        initial = trace.ProxyTracerProvider()
        current = [initial]
        providers = []

        def install(provider):
            self.assertIs(current[0], initial, "FastAPI installed a provider first")
            current[0] = provider
            providers.append(provider)

        def processor(exporter, **kwargs):
            self.assertIsInstance(exporter, PrivateExporter)
            return SimpleSpanProcessor(exporter)

        try:
            with (
                patch.dict(
                    os.environ,
                    {
                        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://collector:4318/v1/traces",
                        "OTEL_SDK_DISABLED": "false",
                        "KADAN_TRACE_SAMPLE_RATIO": "1",
                    },
                    clear=True,
                ),
                patch("api.server.inference", manager),
                patch("opentelemetry.trace.get_tracer_provider", lambda: current[0]),
                patch("opentelemetry.trace.set_tracer_provider", install),
                patch("fastapi.telemetry._runtime._configured", []),
                patch("fastapi.telemetry._runtime._owned", []),
                patch("opentelemetry.sdk.trace.export.BatchSpanProcessor"),
                patch(
                    "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter"
                ) as automatic_exporter,
                patch("kadan_tracing.OTLPSpanExporter", return_value=destination),
                patch("kadan_tracing.BatchSpanProcessor", side_effect=processor),
                patch("kadan_tracing.HTTPXClientInstrumentor"),
                patch("kadan_tracing.set_global_textmap"),
                patch("kadan_tracing._processor"),
            ):
                # FastAPI runs this before the application's lifespan startup.
                _runtime._configure_from_environment(app._telemetry)
                automatic_exporter.assert_not_called()
                self.assertIs(current[0], initial)

                async with lifespan(app):
                    provider = current[0]
                    with provider.get_tracer("unit").start_as_current_span(
                        "private URL",
                        attributes={
                            "http.route": "/items/{item_id}",
                            "url.path": "/items/private",
                            "url.query": "q=private",
                            "prompt": "private prompt",
                            "tool.result": "private result",
                        },
                    ) as span:
                        span.set_status(Status(StatusCode.ERROR, "private error"))
                        span.add_event("private event")

                exported = destination.get_finished_spans()
                self.assertEqual(len(exported), 1)
                self.assertEqual(
                    exported[0].resource.attributes, {"service.name": "hikari-api"}
                )
                self.assertEqual(exported[0].name, "HTTP request")
                self.assertEqual(
                    exported[0].attributes, {"http.route": "/items/{item_id}"}
                )
                self.assertFalse(exported[0].events)
                self.assertIsNone(exported[0].status.description)
                manager.start.assert_awaited_once()
                manager.close.assert_awaited_once()
        finally:
            for provider in providers:
                provider.shutdown()
