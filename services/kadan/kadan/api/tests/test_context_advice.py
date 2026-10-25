import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from kadan.api.config import ServiceError
from kadan.api.context_advice import context_advice
from kadan.api.routes.agents.conversations.runs import RunInput
from kadan.api.routes.status import InferenceState
from kadan.context_budget import context_characters


@pytest.fixture(autouse=True)
def local_file_read(monkeypatch):
    async def read_file(function):
        return function()

    monkeypatch.setattr("kadan.api.context_advice.asyncio.to_thread", read_file)


class Boundary:
    def __init__(self, tmp_path):
        instructions = tmp_path / "AGENT.md"
        instructions.write_text("Shared instructions")
        self.settings = SimpleNamespace(
            agent_instruction_file=instructions, sandbox_url=None
        )
        self.conversation_id = uuid4()
        self.reference_id = uuid4()
        self.workflow_id = uuid4()
        self.resources = {
            self.conversation_id: SimpleNamespace(version=1, kind="conversation"),
            self.reference_id: SimpleNamespace(
                version=2, markdown="Saved reference", kind="memory"
            ),
            self.workflow_id: SimpleNamespace(
                version=3, markdown="Workflow", kind="space", tags=["workflow"]
            ),
        }
        self.status_reads = 0

    async def resource_of_kinds(self, key, resource_id, kinds):
        resource = self.resources[resource_id]
        if resource.kind not in kinds:
            raise ServiceError("Wrong resource kind", 404)
        return resource

    async def inference_status(self):
        self.status_reads += 1
        return InferenceState(
            state="ready",
            message="Ready",
            settings_url="",
            context_limit=8192,
            context_source="effective",
        )

    def input(self, **changes):
        return RunInput(
            **{
                "content": "Current task",
                "expected_version": 1,
                "persona_id": None,
                "references": [
                    {"resource_id": self.reference_id, "expected_version": 2}
                ],
                "workflow_id": self.workflow_id,
                "workflow_version": 3,
            }
            | changes
        )


def test_advice_measures_selected_text_without_sending(tmp_path):
    services = Boundary(tmp_path)
    result = asyncio.run(
        context_advice(services, "alpha", services.conversation_id, services.input())
    )
    assert result.task_characters == len("Current task")
    assert result.instruction_characters == len("Shared instructions")
    assert result.selected_characters == len("Saved referenceWorkflow")
    assert result.tool_schema_characters > 0
    assert result.approximate_character_budget == (8192 - 1024) * 3
    assert result.context_source == "effective"
    assert services.status_reads == 1


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_reference",
        "reference_version",
        "workflow_version",
        "workflow_kind",
        "workflow_tags",
        "reference_length",
        "conversation_version",
        "persona_kind",
    ],
)
def test_invalid_selection_rejected_before_metadata_read(tmp_path, change):
    services = Boundary(tmp_path)
    changes = {}
    if change == "duplicate_reference":
        changes["references"] = services.input().references * 2
    elif change == "reference_version":
        services.resources[services.reference_id].version = 9
    elif change == "workflow_version":
        changes["workflow_version"] = 4
    elif change == "workflow_kind":
        services.resources[services.workflow_id].kind = "memory"
    elif change == "workflow_tags":
        services.resources[services.workflow_id].tags = []
    elif change == "reference_length":
        services.resources[services.reference_id].markdown = "x" * 12001
    elif change == "conversation_version":
        changes["expected_version"] = 2
    elif change == "persona_kind":
        changes["persona_id"] = services.reference_id
    with pytest.raises(ServiceError):
        asyncio.run(
            context_advice(
                services, "alpha", services.conversation_id, services.input(**changes)
            )
        )
    assert services.status_reads == 0


@pytest.mark.parametrize("tokens, expected", [(None, 48000), (1024, 0), (1025, 3)])
def test_approximate_budget_boundary(tokens, expected):
    assert context_characters(tokens) == expected


def test_known_minimum_budget_boundary_and_selected_material_risk(tmp_path):
    services = Boundary(tmp_path)
    value = services.input()
    initial = asyncio.run(
        context_advice(services, "alpha", services.conversation_id, value)
    )
    extra = initial.approximate_character_budget - initial.minimum_characters
    value = value.model_copy(update={"content": value.content + "x" * extra})
    exact = asyncio.run(
        context_advice(services, "alpha", services.conversation_id, value)
    )
    assert not exact.minimum_exceeds_budget
    assert exact.selected_material_risk
    over = asyncio.run(
        context_advice(
            services,
            "alpha",
            services.conversation_id,
            value.model_copy(update={"content": value.content + "x"}),
        )
    )
    assert over.minimum_exceeds_budget
