from datetime import UTC, datetime
from uuid import uuid4

import pytest
from kadan_tracing import carrier
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from tsuchi.models import AgentRun, ImageJob, PetJob


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
