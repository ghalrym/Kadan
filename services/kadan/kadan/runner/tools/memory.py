"""Tools for an agent's own durable memories, separate from Workspace documents."""

from uuid import UUID

from pydantic import Field

from ..memories import AgentMemories
from .base import Arguments, memory_tool


class ListMemoriesArguments(Arguments):
    query: str = Field(default="", max_length=500)
    limit: int = Field(default=20, ge=1, le=100, strict=True)
    offset: int = Field(default=0, ge=0, strict=True)


class ReadMemoryArguments(Arguments):
    memory_id: UUID
    include_trashed: bool = Field(default=False, strict=True)


class CreateMemoryArguments(Arguments):
    title: str = Field(min_length=1, max_length=500, pattern=r"\S")
    markdown: str = Field(max_length=16000)
    tags: list[str] = Field(default_factory=list, max_length=100)


class UpdateMemoryArguments(Arguments):
    memory_id: UUID
    expected_version: int | None = Field(
        default=None,
        ge=1,
        strict=True,
        description="Optional current version for strict conflict checking; omit to update in one call.",
    )
    title: str = Field(min_length=1, max_length=500, pattern=r"\S")
    markdown: str = Field(max_length=16000)
    tags: list[str] | None = Field(default=None, max_length=100)


class ForgetMemoryArguments(Arguments):
    memory_id: UUID
    expected_version: int | None = Field(
        default=None,
        ge=1,
        strict=True,
        description="Optional current version for strict conflict checking; omit to forget in one call.",
    )


@memory_tool("GET")
async def list_memories(args: ListMemoriesArguments, memories: AgentMemories) -> dict:
    """List your active memory summaries, optionally searching stored words.

    Core memories are already loaded automatically into context and relevant
    archival memories are recalled when mentioned; use this tool to search deeper.
    Use offset for more results and read_memory for full content. Memories are
    agent-maintained records the user can view, separate from Workspace documents.
    """
    return await memories.list(args.query, args.limit, args.offset)


@memory_tool("GET")
async def read_memory(args: ReadMemoryArguments, memories: AgentMemories) -> dict:
    """Read one of your active memories, including its current version.

    Set include_trashed to check an uncertain forget operation. Trashed memories
    return identity, version and deleted_at without their content.
    """
    return await memories.read(args.memory_id, args.include_trashed)


@memory_tool("POST")
async def create_memory(args: CreateMemoryArguments, memories: AgentMemories) -> dict:
    """Save a new memory you want to retain beyond recent chat.

    Tag always-needed facts (such as the user's name, standing preferences, or
    major ongoing context) with "core" so they are loaded automatically in future
    conversations and voice calls. Search first to avoid duplicates. This creates
    a memory, not a Workspace document. A failed or interrupted write may have saved;
    check before retrying.
    """
    return await memories.create(args.title, args.markdown, args.tags)


@memory_tool("PUT")
async def update_memory(args: UpdateMemoryArguments, memories: AgentMemories) -> dict:
    """Update a memory's title and Markdown in a single call.

    Omit expected_version to save directly against the latest version, or pass
    expected_version for strict conflict checking. Omit tags to preserve them.
    On a version conflict, read and reconcile before saving again. Other metadata
    stays intact. Check uncertain writes before retrying.
    """
    return await memories.update(
        args.memory_id, args.expected_version, args.title, args.markdown, args.tags
    )


@memory_tool("DELETE")
async def forget_memory(args: ForgetMemoryArguments, memories: AgentMemories) -> dict:
    """Move an outdated memory to Tsuchi trash in a single call.

    Omit expected_version to forget directly against the latest version, or pass
    expected_version for strict conflict checking. Use when the user asks to forget
    a memory. The user can restore it in Tsuchi; this does not erase chat history
    or revisions. After a conflict or uncertain result, read_memory with
    include_trashed before retrying.
    """
    return await memories.forget(args.memory_id, args.expected_version)


TOOLS = [list_memories, read_memory, create_memory, update_memory, forget_memory]
