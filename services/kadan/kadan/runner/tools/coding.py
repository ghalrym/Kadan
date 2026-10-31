from typing import Protocol
from uuid import UUID

from pydantic import Field

from ...api.coding_client import CodingClient
from ...api.pydantic_models.coding import CodexTask
from ...api.sandbox_client import SandboxClient
from .base import Arguments, computer_tool, conversation_tool


class CodexStartArguments(CodexTask):
    message: str = Field(
        min_length=1,
        max_length=2000,
        pattern=r"\S",
        description="Short message telling the user what Codex is doing for them. "
        "It is delivered once Codex has started.",
    )


class CodexConversation(Protocol):
    async def start_codex(self, args: CodexStartArguments) -> dict: ...


@conversation_tool
async def codex_start(
    args: CodexStartArguments, conversation: CodexConversation
) -> dict:
    """Start Codex in the background, tell the user, and end this response.

    Give a new task with an absolute project folder (created if missing), or
    continue an existing thread with thread_id. When Codex finishes, you are
    woken in this conversation with its result.
    """
    return await conversation.start_codex(args)


class CodexThreadsArguments(Arguments):
    thread_id: UUID | None = Field(
        default=None, description="Omit to list recent threads."
    )
    last: int = Field(
        default=10, ge=1, le=50, description="How many recent thread items to read."
    )


@computer_tool("GET")
async def codex_threads(args: CodexThreadsArguments, computer: SandboxClient) -> dict:
    """List Codex threads on your computer, or read one thread's latest messages.

    A thread is still working while running is true.
    """
    client = CodingClient(computer)
    if args.thread_id is None:
        status = await client.status()
        return {
            "setup": status.setup.state,
            "threads": [
                thread.model_dump(mode="json") for thread in status.threads[:10]
            ],
        }
    thread = await client.thread(args.thread_id)
    return {
        **thread.model_dump(mode="json", exclude={"items", "items_truncated"}),
        "items": [item.model_dump() for item in thread.items[-args.last :]],
    }


class CodexStopArguments(Arguments):
    thread_id: UUID


@computer_tool("POST")
async def codex_stop(args: CodexStopArguments, computer: SandboxClient) -> dict:
    """Stop a running Codex thread. File changes it already made stay."""
    return (await CodingClient(computer).stop(args.thread_id)).model_dump(mode="json")


TOOLS = [codex_threads, codex_stop]
START_TOOLS = [codex_start]
