import unittest
from types import SimpleNamespace
from unittest.mock import patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from api.pydantic_models.inference_task import InferenceTask
from api.services.inference_worker import InferenceJob
from api.services.tracing import JobTracing
from api.services.worker_hub import WorkerHub


class JobTracingTests(unittest.TestCase):
    def setUp(self):
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider(shutdown_on_exit=False)
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        tracer = self.provider.get_tracer("unit")
        self.api = patch("api.services.tracing.trace.get_tracer", return_value=tracer)
        self.worker = patch("api.services.tracing.worker_tracer", return_value=tracer)
        self.api.start()
        self.worker.start()
        self.addCleanup(self.api.stop)
        self.addCleanup(self.worker.stop)
        self.addCleanup(self.provider.shutdown)

    def test_native_span_retains_context_and_real_worker_clock(self):
        job = JobTracing()
        worker = SimpleNamespace(kind="llm")
        job.native(
            worker,
            {
                "phase": "prefill",
                "start": 100,
                "end": 200,
                "traceparent": job.parent["traceparent"],
            },
        )
        native = self.exporter.get_finished_spans()[0]
        self.assertEqual(native.parent.span_id, job.span.get_span_context().span_id)
        self.assertEqual(native.start_time, 100)
        self.assertEqual(native.end_time, 200)
        self.assertIs(native.attributes["kadan.kv_cache.cross_request_reuse"], False)
        job.end()

    def test_unrelated_malformed_and_unknown_phases_are_ignored(self):
        job = JobTracing()
        worker = SimpleNamespace(kind="image")
        for message in [
            None,
            {},
            {
                "phase": "prompt contents",
                "start": 100,
                "end": 200,
                "traceparent": job.parent["traceparent"],
            },
            {"phase": "load", "start": 100, "end": 200, "traceparent": "unrelated"},
        ]:
            job.native(worker, message)
        self.assertEqual(self.exporter.get_finished_spans(), ())
        job.end()

    def test_closed_job_rejects_late_spans_and_ends_once(self):
        job = JobTracing()
        job.end(failed=True)
        job.end()
        job.native(
            SimpleNamespace(kind="tts"),
            {
                "phase": "execute",
                "start": 100,
                "end": 200,
                "traceparent": job.parent["traceparent"],
            },
        )
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status.status_code, StatusCode.ERROR)

    def test_duplicate_queue_ack_does_not_reset_wait(self):
        job = JobTracing()
        worker = SimpleNamespace(id="one", kind="image")
        for state, stamp in [("queued", 100), ("queued", 150), ("preparing", 200)]:
            job.event(worker, SimpleNamespace(state=state, trace_unix_ns=stamp))
        wait = self.exporter.get_finished_spans()[0]
        self.assertEqual((wait.start_time, wait.end_time), (100, 200))
        job.end()


class OptionalTelemetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_optional_timestamp_does_not_drop_result(self):
        hub = WorkerHub()
        job = InferenceJob(
            InferenceTask(input={"feature": "image", "prompt": "unit"}), False
        )
        worker = SimpleNamespace(id="one", kind="image", name="unit", controls={})
        job.machines.add(worker.id)
        hub.jobs[job.request.id] = job
        await hub._dispatch(
            worker,
            {
                "id": job.request.id,
                "state": "succeeded",
                "result": {"value": 42},
                "trace_unix_ns": "invalid",
            },
        )
        self.assertTrue(job.result.done())
        self.assertEqual(job.result.result(), {"value": 42})
