"""How an agent tool is declared: a typed argument model and an async function.

The function's docstring is the description the model sees, and its `args`
annotation is the argument schema. Computer tools receive the sandbox client;
conversation tools receive the run's conversation.
"""

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, get_type_hints

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, ValidationError

from ...api.config import ServiceError


class Arguments(BaseModel):
    """Base for argument models; unknown arguments are rejected."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class NoArguments(Arguments):
    pass


def body(args: BaseModel) -> dict[str, Any]:
    """Arguments as a JSON body, leaving unset optional fields to the sandbox."""
    return args.model_dump(mode="json", exclude_none=True)


def register(
    function: Callable[..., Awaitable[Any]], context: str, metadata: dict[str, Any]
) -> StructuredTool:
    schema = get_type_hints(function)["args"]

    async def run(config: RunnableConfig, **arguments: Any) -> Any:
        return await function(schema(**arguments), config["configurable"][context])

    return StructuredTool.from_function(
        coroutine=run,
        name=function.__name__,
        description=inspect.getdoc(function),
        args_schema=schema,
        metadata=metadata,
    )


def computer_tool(
    method: str,
) -> Callable[[Callable[..., Awaitable[Any]]], StructuredTool]:
    """A tool that calls one sandbox route with `method`.

    The method is kept as metadata: after a runner restart only GET calls are
    repeated, because the sandbox has no idempotency keys.
    """
    return lambda function: register(function, "computer", {"method": method})


async def invoke_computer_tool(
    tool: StructuredTool, arguments: dict[str, Any], computer: Any, replayed: bool
) -> dict:
    """Run one computer tool, reporting failures as results rather than errors."""
    if not computer.configured:
        return {"error": "No computer is configured for this agent."}
    if replayed and tool.metadata["method"] != "GET":
        # The sandbox has no idempotency keys, so a restart must not
        # repeat a side effect that may already have happened.
        return {
            "error": "The runner restarted before this operation reported "
            "a result. It may or may not have run; check before retrying."
        }
    try:
        return await tool.ainvoke(
            arguments, config={"configurable": {"computer": computer}}
        )
    except (ServiceError, ValidationError) as error:
        return {"error": str(error)}


def task_tool(function: Callable[..., Awaitable[Any]]) -> StructuredTool:
    """A tool that acts on the computer agent's current task."""
    return register(function, "task", {})


def conversation_tool(function: Callable[..., Awaitable[Any]]) -> StructuredTool:
    """A tool that acts on the conversation: messages, images or the run itself."""
    return register(function, "conversation", {})


def memory_tool(
    method: str,
) -> Callable[[Callable[..., Awaitable[Any]]], StructuredTool]:
    """A scoped memory operation; only reads are safe to replay after a restart."""
    return lambda function: register(function, "memories", {"method": method})
