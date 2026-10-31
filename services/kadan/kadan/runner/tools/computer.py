"""Handing work to the computer agent, and that agent ending its task."""

from typing import Protocol

from pydantic import Field

from .base import Arguments, conversation_tool, task_tool


class UseComputerArguments(Arguments):
    instructions: str = Field(
        min_length=1,
        max_length=10000,
        pattern=r"\S",
        description="What to do on the computer, with any paths, URLs and what "
        "a finished result looks like.",
    )
    message: str = Field(
        min_length=1,
        max_length=2000,
        pattern=r"\S",
        description="Short message telling the user what is happening on the "
        "computer. It is delivered when the task is handed over.",
    )


class ComputerConversation(Protocol):
    async def use_computer(self, args: UseComputerArguments) -> dict: ...


@conversation_tool
async def use_computer(
    args: UseComputerArguments, conversation: ComputerConversation
) -> dict:
    """Hand a task to your computer assistant, tell the user, and end this response.

    The assistant works on the shared computer in the background with browser,
    shell, file and code tools. When it finishes, you are woken in this
    conversation with its summary. Tasks run one at a time in order.
    """
    return await conversation.use_computer(args)


class FinishTaskArguments(Arguments):
    summary: str = Field(
        min_length=1,
        max_length=6000,
        pattern=r"\S",
        description="What you did and found, including paths, URLs and anything "
        "that did not work.",
    )
    succeeded: bool = Field(
        default=True, description="False when the task could not be done."
    )


class ComputerTaskState(Protocol):
    async def finish_task(self, args: FinishTaskArguments) -> dict: ...


@task_tool
async def finish_task(args: FinishTaskArguments, task: ComputerTaskState) -> dict:
    """End the current task and report its result.

    Only finish once the work is done or cannot be done.
    """
    return await task.finish_task(args)


TOOLS = [use_computer]
TASK_TOOLS = [finish_task]
