import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from kadan_tracing import carrier
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

from kadan.api.pydantic_models.agent_runs import AgentRun, ImageJob
from kadan.api.pydantic_models.pets import PetJob
from kadan.runner.image_jobs import ImageJobs
from kadan.runner.pet_jobs import PetJobs


@pytest.mark.parametrize("ratio", [0, 1])
def test_durable_models_accept_actual_sdk_trace_flags(ratio):
    provider = TracerProvider(sampler=TraceIdRatioBased(ratio), shutdown_on_exit=False)
    with trace.use_span(
        provider.get_tracer("unit").start_span("enqueue"), end_on_exit=True
    ):
        parent = carrier()["traceparent"]
    now = datetime.now(UTC)
    fields = dict(
        id=uuid4(),
        agent_key="alpha",
        version=1,
        created_at=now,
        updated_at=now,
        trace_parent=parent,
    )
    assert AgentRun(**fields, conversation_id=uuid4()).trace_parent == parent
    assert (
        ImageJob(
            **fields,
            run_id=uuid4(),
            conversation_id=uuid4(),
            request_id=uuid4(),
            event_run_id=uuid4(),
            prompt="unit",
        ).trace_parent
        == parent
    )
    assert PetJob(**fields, prompt="unit").trace_parent == parent
    provider.shutdown()


@pytest.mark.parametrize("kind", ["image", "pet"])
def test_queue_claim_retains_initial_wait_and_parent(kind):
    provider = TracerProvider(shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("unit")
    with trace.use_span(tracer.start_span("enqueue"), end_on_exit=True):
        parent = carrier()["traceparent"]
    now = datetime.now(UTC)
    queued_at = now - timedelta(seconds=2)
    fields = dict(
        id=uuid4(),
        agent_key="alpha",
        version=1,
        created_at=now - timedelta(days=1),
        updated_at=queued_at,
        trace_parent=parent,
        prompt="unit",
    )
    job = (
        PetJob(**fields)
        if kind == "pet"
        else ImageJob(
            **fields,
            run_id=uuid4(),
            conversation_id=uuid4(),
            request_id=uuid4(),
            event_run_id=uuid4(),
        )
    )

    async def run():
        services = SimpleNamespace(
            json_request=AsyncMock(return_value=[job.model_dump(mode="json")])
        )
        queue = PetJobs(services) if kind == "pet" else ImageJobs(services)

        async def claim(value, **changes):
            return value.model_copy(
                update={
                    "updated_at": datetime.now(UTC),
                    **{key: item for key, item in changes.items() if key != "worker"},
                }
            )

        async def process(value, **kwargs):
            assert value.status == "running"
            assert trace.get_current_span().attributes["kadan.recovery"] is False
            queue.stopping = True

        queue.save = AsyncMock(side_effect=claim)
        queue.process = AsyncMock(side_effect=process)
        await queue.run()
        queue.process.assert_awaited_once()

    with patch("opentelemetry.trace.get_tracer", return_value=tracer):
        asyncio.run(run())
    wait = next(
        item
        for item in exporter.get_finished_spans()
        if item.name == "kadan.queue.wait"
    )
    assert wait.start_time == int(queued_at.timestamp() * 1_000_000_000)
    assert 2_000_000_000 <= wait.end_time - wait.start_time < 60_000_000_000
    provider.shutdown()
