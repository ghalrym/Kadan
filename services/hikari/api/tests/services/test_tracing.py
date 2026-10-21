import unittest
from types import SimpleNamespace
from unittest.mock import patch

from kadan_tracing.cuda_metrics import CUDA_KERNELS
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from api.pydantic_models.inference_task import InferenceTask
from api.services.inference_worker import InferenceJob
from api.services.tracing import JobTracing, TextPhaseMetrics
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
        self.assertNotIn("kadan.kv_cache.cross_request_reuse", native.attributes)
        job.end()

    def native_message(self, job, metrics):
        return {
            "phase": "prefill",
            "start": 100,
            "end": 200,
            "traceparent": job.parent["traceparent"],
            "metrics": metrics,
        }

    def text_metrics(self):
        return {name: 0 for name in TextPhaseMetrics.model_fields}

    def test_text_metrics_are_numeric_attributes_on_existing_span(self):
        job = JobTracing()
        metrics = self.text_metrics()
        metrics.update(tokens=9, large_row_calls=2, large_row_seconds=0.5)
        job.native(SimpleNamespace(kind="llm"), self.native_message(job, metrics))
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].name, "kadan.native.prefill")
        self.assertEqual(spans[0].attributes["kadan.text.tokens"], 9)
        self.assertEqual(spans[0].attributes["kadan.text.large_row_seconds"], 0.5)
        self.assertEqual(len(spans[0].attributes), 27)
        self.assertFalse(spans[0].events)
        job.end()

    def cuda_metrics(self):
        return {
            "kernels": [[0, 0.0, 0, 0.0] for _ in CUDA_KERNELS],
            "copies": [[0, 0.0] for _ in range(5)],
            "sync_calls": 1,
            "sync_seconds": 0.1,
            "dropped": 2,
            "unavailable": 3,
            "shared_batch_size": 2,
        }

    def test_cuda_details_keep_gpu_and_host_times_separate_on_shared_execute(self):
        job = JobTracing()
        cuda = self.cuda_metrics()
        cuda["kernels"][CUDA_KERNELS.index("gguf_matvec_kernel")] = [9, 0.2, 4, 0.3]
        cuda["copies"][4] = [3, 0.4]
        message = self.native_message(job, None) | {"phase": "execute", "cuda": cuda}
        job.native(SimpleNamespace(kind="image"), message)
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        attributes = spans[0].attributes
        self.assertEqual(attributes["kadan.cuda.gguf_matvec_kernel.gpu_seconds"], 0.3)
        self.assertEqual(attributes["kadan.cuda.gguf_matvec_kernel.host_seconds"], 0.2)
        self.assertEqual(attributes["kadan.cuda.gguf_matvec_kernel.measured"], 4)
        self.assertEqual(attributes["kadan.cuda.shared_batch_size"], 2)
        self.assertEqual(attributes["kadan.cuda.copy4.calls"], 3)
        self.assertEqual(attributes["kadan.cuda.dropped"], 2)
        self.assertEqual(attributes["kadan.cuda.unavailable"], 3)
        job.end()

    def test_invalid_cuda_details_do_not_discard_text_details_or_span(self):
        invalids = [None, "private", {"prompt": "private"}]
        for key, bad in [
            ("kernels", [[0, 0.0, 0, 0.0]]),
            ("copies", [[True, 0.0] for _ in range(5)]),
            ("sync_seconds", float("nan")),
            ("sync_seconds", -1),
            ("shared_batch_size", 33),
            ("dropped", 2**63),
        ]:
            invalids.append(self.cuda_metrics() | {key: bad})
        for invalid in invalids:
            job = JobTracing()
            message = self.native_message(job, self.text_metrics()) | {"cuda": invalid}
            job.native(SimpleNamespace(kind="llm"), message)
            span = self.exporter.get_finished_spans()[-1]
            self.assertEqual(span.name, "kadan.native.prefill")
            self.assertIn("kadan.text.tokens", span.attributes)
            self.assertFalse(any(k.startswith("kadan.cuda.") for k in span.attributes))
            job.end()

    def test_invalid_metrics_preserve_the_valid_phase_span(self):
        for invalid in [
            None,
            "private",
            {"prompt": "private"},
            self.text_metrics() | {"tokens": True},
            self.text_metrics() | {"rows": -1},
            self.text_metrics() | {"rows": 2**63},
            self.text_metrics() | {"large_row_seconds": float("nan")},
            self.text_metrics() | {"large_row_seconds": float("inf")},
        ]:
            job = JobTracing()
            job.native(SimpleNamespace(kind="llm"), self.native_message(job, invalid))
            native = self.exporter.get_finished_spans()[-1]
            self.assertEqual(native.name, "kadan.native.prefill")
            self.assertFalse(
                any(key.startswith("kadan.text.") for key in native.attributes)
            )
            job.end()

    def test_text_metrics_are_ignored_for_other_workers(self):
        job = JobTracing()
        job.native(
            SimpleNamespace(kind="image"), self.native_message(job, self.text_metrics())
        )
        self.assertFalse(
            any(
                key.startswith("kadan.text.")
                for key in self.exporter.get_finished_spans()[0].attributes
            )
        )
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
