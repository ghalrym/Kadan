"""Workspace documents and files stored in the current agent's Tsuchi namespace."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from pydantic import Field, field_validator

from .base import Arguments, conversation_tool


class CreateWorkspaceDocumentArguments(Arguments):
    title: str = Field(min_length=1, max_length=500, pattern=r"\S")
    markdown: str = Field(default="", max_length=48000)
    parent_id: UUID | None = Field(
        default=None,
        description="An existing active Space document owned by this agent. "
        "Omit to create a top-level document.",
    )

    @field_validator("title")
    @classmethod
    def strip_title(cls, value: str) -> str:
        return value.strip()


class Workspace(Protocol):
    async def create_workspace_document(
        self, args: CreateWorkspaceDocumentArguments
    ) -> dict: ...

    async def read_workspace_file(self, args: ReadWorkspaceFileArguments) -> dict: ...

    async def list_workspace_files(self, args: ListWorkspaceFilesArguments) -> dict: ...


@conversation_tool
async def create_workspace_document(
    args: CreateWorkspaceDocumentArguments, conversation: Workspace
) -> dict:
    """Create a Markdown Workspace document for this agent.

    Supply parent_id to create a subdocument inside an existing Space. Memories,
    personas, conversations and decks cannot be parents. Returns the saved ID
    for creating further subdocuments. A failed or uncertain result is not
    confirmation that the document was saved; check Workspace before retrying
    an uncertain write.
    """
    return await conversation.create_workspace_document(args)


class ReadWorkspaceFileArguments(Arguments):
    file_id: UUID


class ListWorkspaceFilesArguments(Arguments):
    query: str = Field(default="", max_length=500)
    limit: int = Field(default=20, ge=1, le=50)
    offset: int = Field(default=0, ge=0)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        return value.strip()


@conversation_tool
async def read_workspace_file(
    args: ReadWorkspaceFileArguments, conversation: Workspace
) -> dict:
    """Read saved file metadata and its download URL.

    This does not read file contents. Missing or trashed files are reported as
    errors.
    """
    return await conversation.read_workspace_file(args)


@conversation_tool
async def list_workspace_files(
    args: ListWorkspaceFilesArguments, conversation: Workspace
) -> dict:
    """List or search this agent's active file metadata, newest update first, then ID.

    Optional query requires all whitespace-separated tokens to match
    case-insensitive substrings in title, filename, tags or source labels/URLs.
    Never searches contents or meaning; empty files means no metadata matches.
    Names can repeat; choose by file_id, then read_workspace_file. Continue
    with next_offset and the same query/limit; null ends paging. Concurrent
    edits can shift pages. No file bytes.
    """
    return await conversation.list_workspace_files(args)


TOOLS = [
    create_workspace_document,
    read_workspace_file,
    list_workspace_files,
]
