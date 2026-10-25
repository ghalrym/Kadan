"""Read-only draft measurements using the runner's existing character heuristic."""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from pydantic import BaseModel

from ..context_budget import context_characters
from ..runner.agent_inference import tool_characters
from ..runner.tools import request_tools
from .config import ServiceError
from .pydantic_models.resources import ResourceKind

if TYPE_CHECKING:
    from .routes.agents.conversations.runs import RunInput
    from .services import LocalServices


class ContextAdvice(BaseModel):
    model_id: str | None
    context_limit: int | None
    context_source: Literal["effective", "configured", "architecture", "unknown"]
    approximate_character_budget: int
    task_characters: int
    instruction_characters: int
    selected_characters: int
    tool_schema_characters: int
    minimum_characters: int
    minimum_exceeds_budget: bool
    selected_material_risk: bool


async def context_advice(
    services: "LocalServices", key: str, conversation_id: UUID, value: "RunInput"
) -> ContextAdvice:
    conversation = await services.resource_of_kinds(
        key, conversation_id, (ResourceKind.CONVERSATION,)
    )
    if conversation.version != value.expected_version:
        raise ServiceError("Conversation changed. Reload before checking context.", 409)
    try:
        instructions = await asyncio.to_thread(
            Path(services.settings.agent_instruction_file).read_text
        )
    except (OSError, UnicodeError) as error:
        raise ServiceError(
            "Shared agent instructions could not be read.", 503
        ) from error
    if value.persona_id:
        persona = await services.resource_of_kinds(
            key, value.persona_id, (ResourceKind.PERSONA,)
        )
        instructions += "\nSelected persona:\n" + persona.markdown
    if len({selection.resource_id for selection in value.references}) != len(
        value.references
    ):
        raise ServiceError("Select each reference only once.", 422)
    selected_characters = 0
    for selection in value.references:
        resource = await services.resource_of_kinds(
            key, selection.resource_id, (ResourceKind.SPACE, ResourceKind.MEMORY)
        )
        if resource.version != selection.expected_version:
            raise ServiceError("Selected reference changed. Select it again.", 409)
        selected_characters += len(resource.markdown)
    if selected_characters > 12000:
        raise ServiceError("Selected references exceed 12000 characters", 422)
    if value.workflow_id:
        workflow = await services.resource_of_kinds(
            key, value.workflow_id, (ResourceKind.SPACE,)
        )
        if (
            workflow.version != value.workflow_version
            or "workflow" not in workflow.tags
        ):
            raise ServiceError("Selected workflow changed. Select it again.", 409)
        if not workflow.markdown.strip() or len(workflow.markdown) > 12000:
            raise ServiceError(
                "Workflow instructions must contain 1–12000 characters", 409
            )
        selected_characters += len(workflow.markdown)
    schema_characters = tool_characters(
        request_tools(
            sandbox_configured=bool(services.settings.sandbox_url),
            has_references=bool(value.references),
            has_workflow=value.workflow_id is not None,
        )
    )
    status = await services.inference_status()
    budget = context_characters(status.context_limit)
    minimum = len(value.content) + len(instructions) + schema_characters
    return ContextAdvice(
        model_id=status.model_id,
        context_limit=status.context_limit,
        context_source=status.context_source,
        approximate_character_budget=budget,
        task_characters=len(value.content),
        instruction_characters=len(instructions),
        selected_characters=selected_characters,
        tool_schema_characters=schema_characters,
        minimum_characters=minimum,
        minimum_exceeds_budget=minimum > budget,
        selected_material_risk=minimum + selected_characters > budget,
    )
