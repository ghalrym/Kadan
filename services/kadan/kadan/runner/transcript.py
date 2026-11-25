"""Project saved actions, observations and authored turns onto the model protocol."""

import json

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ..api.config import ServiceError
from ..api.pydantic_models.agent_runs import (
    MigratedMarker,
    RunCall,
    RunMessage,
    RunObservation,
    RunResult,
    TaskEnd,
    TaskHistory,
    TaskStart,
    TranscriptEntry,
    UnrecoveredRecord,
)


def entry_characters(item: TranscriptEntry) -> int:
    if isinstance(item, RunMessage) and not item.attachment_ids:
        return len(item.content)
    if isinstance(item, (MigratedMarker, UnrecoveredRecord)):
        return 0
    # Allow for the call and result headers when projecting a structured observation.
    return len(item.model_dump_json()) + 96


def transcript_messages(transcript: list[TranscriptEntry]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    pending: RunCall | None = None
    for index, item in enumerate(transcript):
        if isinstance(item, (MigratedMarker, UnrecoveredRecord)):
            continue
        if isinstance(item, RunResult):
            if pending is None or pending.call_id != item.call_id:
                raise ServiceError("The saved tool result has no matching action.", 422)
            messages.append(
                ToolMessage(
                    content=json.dumps(item.data, ensure_ascii=False),
                    name=pending.command.tool,
                    tool_call_id=item.call_id,
                )
            )
            pending = None
            continue
        if pending is not None:
            raise ServiceError("A saved action is missing its result.", 422)
        if isinstance(item, RunMessage):
            author = {
                "user": HumanMessage,
                "assistant": AIMessage,
                "system": SystemMessage,
            }
            messages.append(author[item.role](content=item.content))
            if not item.attachment_ids:
                continue
            name = "message_attachments"
            data = {"attachment_ids": [str(asset) for asset in item.attachment_ids]}
        elif isinstance(item, RunCall):
            pending = item
            messages.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": item.command.tool,
                            "args": item.command.arguments,
                            "id": item.call_id,
                        }
                    ],
                )
            )
            continue
        elif isinstance(item, TaskStart) and item.origin == "user":
            messages.append(HumanMessage(content=item.instructions))
            continue
        elif isinstance(item, RunObservation):
            name, data = item.name, item.data
        elif isinstance(item, TaskStart):
            name, data = (
                "read_computer_task",
                item.model_dump(mode="json", exclude={"role"}),
            )
        elif isinstance(item, TaskEnd):
            name, data = (
                "read_computer_task_outcome",
                item.model_dump(mode="json", exclude={"role"}),
            )
        else:
            assert isinstance(item, TaskHistory)
            name, data = (
                "read_computer_task_history",
                item.model_dump(mode="json")["tasks"],
            )
        # The runner obtained this information on the agent's behalf. An atomic
        # call/result pair also represents a failed validation without retaining
        # the rejected arguments.
        call_id = f"observation_{index}"
        messages.extend(
            (
                AIMessage(
                    content="", tool_calls=[{"name": name, "args": {}, "id": call_id}]
                ),
                ToolMessage(
                    content=json.dumps(data, ensure_ascii=False),
                    name=name,
                    tool_call_id=call_id,
                ),
            )
        )
    return messages
